#include <vector>
#include <string>
#include <utility>

#include "caffe/layers/repulsion_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

// bottom[0]: proposal_bbox     bottom[1]: bbox_targets
// bottom[2]: bbox_loss_weights     bottom[3]: RepGT_anchors
// bottom[4]: RepGT_gt      bottom[5]: RepGT_label

namespace caffe{
    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
            has_weights_ = (bottom.size() == 8);
        }

    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
            LossLayer<Dtype>::Reshape(bottom,top);


            //repulsion loss Attr part
            CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
            CHECK_EQ(bottom[0]->width(),bottom[1]->width());
            CHECK_EQ(bottom[0]->height(),bottom[1]->height());
            if(has_weights_){
                CHECK_EQ(bottom[0]->channels(),bottom[2]->channels());
                CHECK_EQ(bottom[0]->width(),bottom[2]->width());
                CHECK_EQ(bottom[0]->height(),bottom[2]->height());
            }

            diff_Attr_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());

            errors_Attr_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());

            

            //repulsion loss RepGT part
            CHECK_EQ(bottom[3]->channels(),bottom[0]->channels());
            CHECK_EQ(bottom[3]->width(),bottom[0]->width());
            CHECK_EQ(bottom[3]->height(),bottom[0]->height());

            CHECK_EQ(bottom[4]->channels(),bottom[0]->channels());
            CHECK_EQ(bottom[4]->width(),bottom[0]->width());
            CHECK_EQ(bottom[4]->height(),bottom[0]->height());

            CHECK_EQ(bottom[5]->channels(),bottom[0]->channels()/4);
            CHECK_EQ(bottom[5]->width(),bottom[0]->width());
            CHECK_EQ(bottom[5]->height(),bottom[0]->height());

            diff_RepGT_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());
            errors_RepGT_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());
          
            
            //repulsion loss RepBox part
            diff_RepBox_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
            errors_RepBox_.Reshape(1, 1, bottom[0]->count() / 4,bottom[0]->count() / 4);
            iou_RepBox_.Reshape(1, 1, bottom[0]->count() / 4, bottom[0]->count() / 4);
            nzeros_RepBox_.Reshape(1,1,1,1);

         

            diff_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());
            errors_.Reshape(bottom[0]->num(),bottom[0]->channels(),
                bottom[0]->height(),bottom[0]->width());
        }
    
    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
            NOT_IMPLEMENTED;
        }
    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
            NOT_IMPLEMENTED;
        }

    #ifdef CPU_ONLY
    STUB_GPU(RepulsionLossLayer);
    #endif

    INSTANTIATE_CLASS(RepulsionLossLayer);
    REGISTER_LAYER_CLASS(RepulsionLoss);
}