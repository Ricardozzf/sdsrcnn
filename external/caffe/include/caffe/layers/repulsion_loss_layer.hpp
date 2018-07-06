#ifndef CAFFE_REPULSION_LOSS_HPP_
#define CAFFE_REPULSION_LOSS_HPP_

#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
    template<typename Dtype>
    class RepulsionLossLayer : public LossLayer<Dtype>{
    public:
        explicit RepulsionLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param){}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const {return "RepulsionLoss";}

        virtual inline int ExactNumBottomBlobs() const {return -1;}

        virtual inline int MinBottomBlobs() const {return 2;}

        virtual inline int MaxBottomBlobs() const {return 8;}

        virtual inline bool AllowForceBackward(const int bottom_index) const{
            return true;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>&  propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
        // Attr blob
        Blob<Dtype> diff_Attr_;
        Blob<Dtype> errors_Attr_;
     

        // RepGT blob
        Blob<Dtype> diff_RepGT_;
        Blob<Dtype> errors_RepGT_;


        // RepBox blob
        Blob<Dtype> diff_RepBox_;
        Blob<Dtype> errors_RepBox_;
        Blob<Dtype> iou_RepBox_;
        Blob<Dtype> nzeros_RepBox_;
     


        bool has_weights_;

        Blob<Dtype> diff_;
        Blob<Dtype> errors_;

    };
} //namespace caffe

#endif