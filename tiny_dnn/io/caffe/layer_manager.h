/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "caffe.pb.h"

namespace tiny_dnn {
namespace detail {

struct layer_node {
    const caffe::LayerParameter *layer;
    std::vector<const layer_node*> next;  // top-side
    std::vector<const layer_node*> prev;  // bottom-side

    layer_node() : layer(0), next(0), prev(0) {}
    explicit layer_node(const caffe::LayerParameter *l)
        : layer(l), next(0), prev(0) {}
};

// parse caffe net and interpret as single layer vector
class caffe_layer_manager {
 public:
    explicit caffe_layer_manager(const caffe::NetParameter& net_orig)
            : net(net_orig) {
        if (net.layers_size() > 0) {
            upgradev1net(net_orig, &net);
        }

        /* blob name -> producer */
        std::map<std::string, layer_node*> blob_producer_layers;


        nodes.reserve(net.layer_size());

        // register layers to layer_table
        for (int i = 0; i < net.layer_size(); i++) {
            auto& l = net.layer(i);

            if (layer_table.find(l.name()) != layer_table.end()) continue;

            nodes.emplace_back(&l);
            layer_table[l.name()] = &nodes.back();
        }

        for (int i = 0; i < nodes.size(); i++) {
            auto& l = nodes[i];

            // bottom
            if (l.layer->bottom_size() > 0) {
                for (size_t j = 0; j < l.layer->bottom_size(); j++) {
                    auto& bottom_blob = l.layer->bottom(j);
                    layer_node* producer = blob_producer_layers[bottom_blob];

                    if (net_orig.input_size() && net_orig.input(0) == bottom_blob) {
                        continue;
                    }

                    if (producer == nullptr) {
                        throw nn_error("unsupported caffemodel: producer of blob[" + bottom_blob + "] not found");
                    }

                    l.prev.push_back(producer);
                    producer->next.push_back(&l);
                }
            }

            // top
            if (l.layer->top_size() > 0) {
                for (size_t j = 0; j < l.layer->top_size(); j++) {
                    blob_producer_layers[l.layer->top(j)] = &l;
                }
            }

            // register input shape
            if (l.layer->has_input_param()) {
                int depth = l.layer->input_param().shape(0).dim(1);
                int height = l.layer->input_param().shape(0).dim(2);
                int width = l.layer->input_param().shape(0).dim(3);

                assert(l.layer->top_size() > 0);
                inferred_shape[l.layer->top(0)] = shape3d(width, height, depth);
            }
        }
    }

    size_t size() const {
        return nodes.size();
    }

    shape3d bottom_shape(const caffe::LayerParameter& layer, size_t idx = 0) const {
        if (layer.bottom_size() <= idx) {
            throw nn_error("invalid top idx");
        }
        auto result = inferred_shape.find(layer.bottom(idx));
        if (result == inferred_shape.end()) {
            throw nn_error("unknown shape");
        }
        return result->second;
    }

    void register_top_shape(const caffe::LayerParameter& layer, const shape3d& shape, size_t idx = 0) {
        if (layer.top_size() <= idx) {
            throw nn_error("invalid bottom idx");
        }
        inferred_shape[layer.top(idx)] = shape;
    }

    void register_top_shape(const std::string& blob_name, const shape3d& shape) {
        inferred_shape[blob_name] = shape;
    }

    const caffe::LayerParameter& operator[] (size_t index) const {
        return *nodes[index].layer;
    }

    void construct_graph(std::unordered_map < const caffe::LayerParameter*, std::shared_ptr<layer>>& caffe2tiny, network<graph>& net) {
        std::vector<std::shared_ptr<layer>> inputs, outputs, all;

        for (size_t i = 0; i < nodes.size(); i++) {
            auto node = nodes[i];

            if (caffe2tiny.find(node.layer) == caffe2tiny.end()) {
                continue;
            }

            auto tiny_node = caffe2tiny.at(node.layer);

            if (node.prev.empty()) {
                inputs.push_back(tiny_node);
            }
            if (node.next.empty()) {
                outputs.push_back(tiny_node);
            }
            all.push_back(tiny_node);

            auto tail = caffe2tiny.at(node.layer);

            for (size_t j = 0; j < nodes[i].prev.size(); j++) {
                if (caffe2tiny.find(node.prev[j]->layer) == caffe2tiny.end()) {
                    continue;
                }
                auto head = caffe2tiny.at(node.prev[j]->layer);
                connect(&*head, &*tail, j, 0);
            }
        }
        assign_nodes(net, all);
        tiny_dnn::construct_graph(net, inputs, outputs);
    }

 private:

    void upgradev1net(const caffe::NetParameter& old,
                      caffe::NetParameter *dst) const {
        dst->CopyFrom(old);
        dst->clear_layers();
        dst->clear_layer();

        for (int i = 0; i < old.layers_size(); i++) {
            upgradev1layer(old.layers(i), dst->add_layer());
        }
    }

    const char* v1type2name(caffe::V1LayerParameter_LayerType type) const {
        switch (type) {
        case caffe::V1LayerParameter_LayerType_NONE:
            return "";
        case caffe::V1LayerParameter_LayerType_ABSVAL:
            return "AbsVal";
        case caffe::V1LayerParameter_LayerType_ACCURACY:
            return "Accuracy";
        case caffe::V1LayerParameter_LayerType_ARGMAX:
            return "ArgMax";
        case caffe::V1LayerParameter_LayerType_BNLL:
            return "BNLL";
        case caffe::V1LayerParameter_LayerType_CONCAT:
            return "Concat";
        case caffe::V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
            return "ContrastiveLoss";
        case caffe::V1LayerParameter_LayerType_CONVOLUTION:
            return "Convolution";
        case caffe::V1LayerParameter_LayerType_DECONVOLUTION:
            return "Deconvolution";
        case caffe::V1LayerParameter_LayerType_DATA:
            return "Data";
        case caffe::V1LayerParameter_LayerType_DROPOUT:
            return "Dropout";
        case caffe::V1LayerParameter_LayerType_DUMMY_DATA:
            return "DummyData";
        case caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
            return "EuclideanLoss";
        case caffe::V1LayerParameter_LayerType_ELTWISE:
            return "Eltwise";
        case caffe::V1LayerParameter_LayerType_EXP:
            return "Exp";
        case caffe::V1LayerParameter_LayerType_FLATTEN:
            return "Flatten";
        case caffe::V1LayerParameter_LayerType_HDF5_DATA:
            return "HDF5Data";
        case caffe::V1LayerParameter_LayerType_HDF5_OUTPUT:
            return "HDF5Output";
        case caffe::V1LayerParameter_LayerType_HINGE_LOSS:
            return "HingeLoss";
        case caffe::V1LayerParameter_LayerType_IM2COL:
            return "Im2col";
        case caffe::V1LayerParameter_LayerType_IMAGE_DATA:
            return "ImageData";
        case caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS:
            return "InfogainLoss";
        case caffe::V1LayerParameter_LayerType_INNER_PRODUCT:
            return "InnerProduct";
        case caffe::V1LayerParameter_LayerType_LRN:
            return "LRN";
        case caffe::V1LayerParameter_LayerType_MEMORY_DATA:
            return "MemoryData";
        case caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
            return "MultinomialLogisticLoss";
        case caffe::V1LayerParameter_LayerType_MVN:
            return "MVN";
        case caffe::V1LayerParameter_LayerType_POOLING:
            return "Pooling";
        case caffe::V1LayerParameter_LayerType_POWER:
            return "Power";
        case caffe::V1LayerParameter_LayerType_RELU:
            return "ReLU";
        case caffe::V1LayerParameter_LayerType_SIGMOID:
            return "Sigmoid";
        case caffe::V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
            return "SigmoidCrossEntropyLoss";
        case caffe::V1LayerParameter_LayerType_SILENCE:
            return "Silence";
        case caffe::V1LayerParameter_LayerType_SOFTMAX:
            return "Softmax";
        case caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS:
            return "SoftmaxWithLoss";
        case caffe::V1LayerParameter_LayerType_SPLIT:
            return "Split";
        case caffe::V1LayerParameter_LayerType_SLICE:
            return "Slice";
        case caffe::V1LayerParameter_LayerType_TANH:
            return "TanH";
        case caffe::V1LayerParameter_LayerType_WINDOW_DATA:
            return "WindowData";
        case caffe::V1LayerParameter_LayerType_THRESHOLD:
            return "Threshold";
        default:
            throw nn_error("unknown v1 layer-type");
        }
    }

    void upgradev1layer(const caffe::V1LayerParameter& old,
                        caffe::LayerParameter *dst) const {
        dst->Clear();

        for (int i = 0; i < old.bottom_size(); i++) {
            dst->add_bottom(old.bottom(i));
        }

        for (int i = 0; i < old.top_size(); i++) {
            dst->add_top(old.top(i));
        }

        if (old.has_name()) dst->set_name(old.name());
        if (old.has_type()) dst->set_type(v1type2name(old.type()));

        for (int i = 0; i < old.blobs_size(); i++) {
            dst->add_blobs()->CopyFrom(old.blobs(i));
        }

        for (int i = 0; i < old.param_size(); i++) {
            while (dst->param_size() <= i) dst->add_param();
            dst->mutable_param(i)->set_name(old.param(i));
        }

        #define COPY_PARAM(name) if (old.has_##name##_param()) dst->mutable_##name##_param()->CopyFrom(old.name##_param())

        COPY_PARAM(accuracy);
        COPY_PARAM(argmax);
        COPY_PARAM(concat);
        COPY_PARAM(contrastive_loss);
        COPY_PARAM(convolution);
        COPY_PARAM(data);
        COPY_PARAM(dropout);
        COPY_PARAM(dummy_data);
        COPY_PARAM(eltwise);
        COPY_PARAM(exp);
        COPY_PARAM(hdf5_data);
        COPY_PARAM(hdf5_output);
        COPY_PARAM(hinge_loss);
        COPY_PARAM(image_data);
        COPY_PARAM(infogain_loss);
        COPY_PARAM(inner_product);
        COPY_PARAM(lrn);
        COPY_PARAM(memory_data);
        COPY_PARAM(mvn);
        COPY_PARAM(pooling);
        COPY_PARAM(power);
        COPY_PARAM(relu);
        COPY_PARAM(sigmoid);
        COPY_PARAM(softmax);
        COPY_PARAM(slice);
        COPY_PARAM(tanh);
        COPY_PARAM(threshold);
        COPY_PARAM(window_data);
        COPY_PARAM(transform);
        COPY_PARAM(loss);
        #undef COPY_PARAM
    }

    caffe::NetParameter net;
    layer_node *root_node;
    /* layer name -> layer */
    std::map<std::string, layer_node*> layer_table;

    std::vector<layer_node> nodes;
    std::vector<const caffe::LayerParameter*> node_list;

    /* blob name -> shape */
    std::map<std::string, shape3d> inferred_shape;
};

}  // namespace detail
}  // namespace tiny_dnn
