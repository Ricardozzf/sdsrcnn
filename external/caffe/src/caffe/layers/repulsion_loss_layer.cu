#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/layers/repulsion_loss_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

#define debug 0

namespace caffe{
    template<typename Dtype>
    __global__ void SmoothL1ForwardGPU(const int n, const Dtype* in, Dtype* out){
        //f(x) = 0.5 * x^2  if |x| < 1
        //       |x| - 0.5  otherwise
        CUDA_KERNEL_LOOP(index, n){
            Dtype val  = in[index];
            Dtype abs_val = abs(val);
            if(abs_val < 1){
                out[index] = 0.5 * val * val;
            }
            else{
                out[index] = abs_val - 0.5;  
            }
        } 
    }

    template<typename Dtype>
    __global__ void SmoothLnForwardGPU(const int n, const Dtype* in, Dtype* out){
        //f(x) = -ln(1-x)     x <= sigma
        //     = (x-sigma)/(1-sigma) - ln(1-sigma)  otherwise
        // according to the repulsion paper parameter sigma  set to 0.5
        Dtype sigma = 0.5;
        CUDA_KERNEL_LOOP(index,n){
            Dtype val = in[index];
            if(val <= sigma){
                out[index] = -log(1 - val);
            }
            else{
                out[index] = (val - sigma) / (1 - sigma) - log(1 - sigma);
            }
        }
    }

    template<typename Dtype>
    __global__ void IoGForwardGPU(const int n, const int* dim, const Dtype* in_p, const Dtype* in_a, 
        const Dtype* in_g,const Dtype* in_l, const Dtype* in_nm, Dtype* out){
        // dim: 1 9 45 60
        // n: 24300

        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num,c,h,w;
            int index_tmp = index;
            num = index_tmp / (C * H * W);
            index_tmp = index_tmp % (C * H * W);

            c = index_tmp / (H * W);
            index_tmp = index_tmp % (H * W);

            h = index_tmp / W;
            index_tmp = index_tmp % W;
            w = index_tmp;


            Dtype tx = in_p[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype ty = in_p[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype tw = in_p[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype th = in_p[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];

            tx = tx * in_nm[4] + in_nm[0];
            ty = ty * in_nm[5] + in_nm[1];
            tw = tw * in_nm[6] + in_nm[2];
            th = th * in_nm[7] + in_nm[3];

            Dtype px = in_a[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype py = in_a[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype pw = in_a[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype ph = in_a[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];

            Dtype Mx = in_g[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype My = in_g[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype Mw = in_g[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype Mh = in_g[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];

            Dtype Gx = px + tx * pw;
            Dtype Gy = py + ty * ph;
            Dtype Gw = pw * exp(tw);
            Dtype Gh = ph * exp(th);

            Dtype start_x = min(Gx, Mx);
            Dtype end_x = max(Gx + Gw, Mx + Mw);
            Dtype start_y = min(Gy, My);
            Dtype end_y = max(Gy + Gh, My + Mh);

            Dtype width_area = Gw + Mw - (end_x - start_x);
            Dtype height_area = Gh + Mh - (end_y - start_y);
            Dtype area = width_area * height_area;
            Dtype area2 = Mw * Mh;
            Dtype ratio;

            if(width_area <=2 || height_area <=2 || area2 == 0){
                ratio = 0;
            }
            else{
                ratio = area / area2;
            }
            
            out[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w] = ratio;
            out[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w] = ratio;
            out[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w] = ratio;
            out[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w] = ratio;

        }   
        
    }

    template<typename Dtype>
    __global__ void ProIoUForwardGPU(const int n, const int* dim, const Dtype* in_p, const Dtype* in_a, const Dtype* in_l,
        const Dtype* in_nm, Dtype* out_n, Dtype* out){
        //n: 24300*24300
        //out is a matrix with 24300*24300
        
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num1,c1,h1,w1;
            int num2,c2,h2,w2;
            int row,col;
            const Dtype val = 1;

            row = index / (C * H * W);
            col = index % (C * H * W); 

            int index1 = row;
            int index2 = col;
            
            num1 = index1 / (C * H * W);
            num2 = index2 / (C * H * W);
            index1 = index1 % (C * H * W);
            index2 = index2 % (C * H * W);

            c1 = index1 / (H * W);
            c2 = index2 / (H * W);
            index1 = index1 % (H * W);
            index2 = index2 % (H * W);

            h1 = index1 / W;
            h2 = index2 / W;
            index1 = index1 % W;
            index2 = index2 % W;
            w1 = index1;
            w2 = index2;

            int l1 = in_l[row];
            int l2 = in_l[col];

            if(l1 == 0 || l2 == 0 || l1 == l2){
                out[row * C * H * W + col] = 0;
                if(l1 != l2){
                    caffe_gpu_atomic_add(val,out_n);
                }
            }
            else{
                Dtype tx1 = in_p[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype ty1 = in_p[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype tw1 = in_p[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype th1 = in_p[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
                tx1 = tx1 * in_nm[4] + in_nm[0];
                ty1 = ty1 * in_nm[5] + in_nm[1];
                tw1 = tw1 * in_nm[6] + in_nm[2];
                th1 = th1 * in_nm[7] + in_nm[3];
    
                Dtype px1 = in_a[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype py1 = in_a[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype pw1 = in_a[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype ph1 = in_a[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype Gx1 = px1 + tx1 * pw1;
                Dtype Gy1 = py1 + ty1 * ph1;
                Dtype Gw1 = pw1 * exp(tw1);
                Dtype Gh1 = ph1 * exp(th1);

                Dtype tx2 = in_p[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype ty2 = in_p[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype tw2 = in_p[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype th2 = in_p[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
                tx2 = tx2 * in_nm[4] + in_nm[0];
                ty2 = ty2 * in_nm[5] + in_nm[1];
                tw2 = tw2 * in_nm[6] + in_nm[2];
                th2 = th2 * in_nm[7] + in_nm[3];
    
                Dtype px2 = in_a[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype py2 = in_a[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype pw2 = in_a[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype ph2 = in_a[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype Gx2 = px2 + tx2 * pw2;
                Dtype Gy2 = py2 + ty2 * ph2;
                Dtype Gw2 = pw2 * exp(tw2);
                Dtype Gh2 = ph2 * exp(th2);


                Dtype start_x = min(Gx1, Gx2);
                Dtype end_x = max(Gx1 + Gw1, Gx2 + Gw2);
                Dtype start_y = min(Gy1, Gy2);
                Dtype end_y = max(Gy1 + Gh1, Gy2 + Gh2);

                Dtype area,width_area,height_area;
                Dtype ratio;
                width_area = Gw1 + Gw2 - (end_x - start_x);
                height_area = Gh1 + Gh2 - (end_y - start_y);

                if(width_area <= 5 || height_area <=5){
                    ratio = 0;
                }
                else{
                    area = width_area * height_area;
                    ratio = area / (Gw1 * Gh1 + Gw2 * Gh2 - area);
                    caffe_gpu_atomic_add(val,out_n);
                }

                out[row * C * H * W + col] = ratio;
                
            }
            
            
        }
    }

    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
            // bottom[0]: proposal_bbox         bottom[1]: bbox_targets
            // bottom[2]: bbox_loss_weights     bottom[3]: RepGT_anchors
            // bottom[4]: RepGT_gt              bottom[5]: RepGT_label
            // bottom[6]: RepBox_label
            
            //Attr
            int count = bottom[0]->count();
            int num_anchor = count / 4;   
            
            caffe_gpu_sub(count,bottom[0]->gpu_data(),bottom[1]->gpu_data(),
                diff_Attr_.mutable_gpu_data());

            if(has_weights_){
                caffe_gpu_mul(count,bottom[2]->gpu_data(),diff_Attr_.gpu_data(),
                    diff_Attr_.mutable_gpu_data());
            }


            SmoothL1ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
                    count,diff_Attr_.gpu_data(),errors_Attr_.mutable_gpu_data());
            CUDA_POST_KERNEL_CHECK;

            // Attr loss
            Dtype loss_Attr;

            caffe_gpu_asum(count, errors_Attr_.gpu_data(), &loss_Attr);
            
            int spatial_dim_Attr = diff_Attr_.height() * diff_Attr_.width();
            loss_Attr = loss_Attr / bottom[0]->num() / spatial_dim_Attr;

            top[0]->mutable_cpu_data()[0] = loss_Attr;

            if(debug){
                std::cout<<"loss_Attr is : "<<loss_Attr<<std::endl;
            }
   
            //RepGT
            IoGForwardGPU<Dtype> <<<CAFFE_GET_BLOCKS(num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor,bottom[5]->gpu_shape(),
                bottom[0]->gpu_data(), bottom[3]->gpu_data(), bottom[4]->gpu_data(),bottom[5]->gpu_data(),bottom[7]->gpu_data(),
                diff_RepGT_.mutable_gpu_data());
            CUDA_POST_KERNEL_CHECK;
            
            SmoothLnForwardGPU<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                diff_RepGT_.gpu_data(),errors_RepGT_.mutable_gpu_data());
            CUDA_POST_KERNEL_CHECK;

            //RepGT loss
            Dtype loss_RepGT;
            caffe_gpu_asum(count, errors_RepGT_.gpu_data(), &loss_RepGT);
            int spatial_dim_RepGT = errors_RepGT_.height() * errors_RepGT_.width();
            loss_RepGT = loss_RepGT / spatial_dim_RepGT / bottom[0]->num();

            top[0]->mutable_cpu_data()[0] += 0 * loss_RepGT;
            if(debug){
                std::cout<<"loss_RepGT is: "<<loss_RepGT<<std::endl;
            }

            //RepBox
            Dtype* p_nzeros_RepBox = nzeros_RepBox_.mutable_cpu_data();
            (*p_nzeros_RepBox) = 0;

            ProIoUForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num_anchor * num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor * num_anchor, bottom[6]->gpu_shape(),
                bottom[0]->gpu_data(),bottom[3]->gpu_data(),bottom[6]->gpu_data(),bottom[7]->gpu_data(),nzeros_RepBox_.mutable_gpu_data(),
                iou_RepBox_.mutable_gpu_data());
            CUDA_POST_KERNEL_CHECK;

            SmoothLnForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num_anchor*num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor * num_anchor, 
                iou_RepBox_.gpu_data(),errors_RepBox_.mutable_gpu_data());
            CUDA_POST_KERNEL_CHECK;
            
   
            Dtype loss_RepBox;
            int nzeros_scal;
            if(nzeros_RepBox_.data_at(0,0,0,0)>0)
                nzeros_scal = nzeros_RepBox_.data_at(0,0,0,0);
            else{
                nzeros_scal = 1;
            }
            caffe_gpu_asum(num_anchor * num_anchor, errors_RepBox_.gpu_data(), &loss_RepBox);
           
            loss_RepBox = loss_RepBox / nzeros_scal;

            int spatial_dim_RepBox = errors_RepBox_.height() * errors_RepBox_.width();
            top[0]->mutable_cpu_data()[0] += 0 * loss_RepBox;
            if(debug){
                std::cout<<"loss_RepBox is: "<<loss_RepBox<<std::endl;
                std::cout<<"loss is: "<<top[0]->mutable_cpu_data()[0]<<std::endl;
                std::cout<<"num of RepBox is:"<<nzeros_RepBox_.data_at(0,0,0,0)<<std::endl;
            }
            

    }

    template<typename Dtype>
    __global__ void SmoothL1BackwardGPU(const int n,const Dtype* in, Dtype* out){
        // f'(x) = x            if |x| < 1
        //       = sign(x)      otherwise
        CUDA_KERNEL_LOOP(index,n){
            Dtype val = in[index];
            Dtype abs_val = abs(val);
            if(abs_val < 1){
                out[index] = val;
            }
            else{
                out[index] = (Dtype(0) < val) - (val < Dtype(0));
            }
        }
    }

    template<typename Dtype>
    __global__ void SmoothLnBackwardGPU(const int n, const Dtype* in, Dtype* out){
        //f'(x) = 1 / (1 - x)          x <= sigma
        //      = 1 / (1 - sigma)      otherwise
        float sigma = 0.5;
        CUDA_KERNEL_LOOP(index,n){
            Dtype val = in[index];
            if(val <= sigma){
                out[index] = 1 / (1 - val);
            }
            else{
                out[index] = 1 / (1 - sigma);
            }
        }
    }

    template<typename Dtype>
    __global__ void RepGTBackwardGPU(const int n,const int* dim, const Dtype* in_iog, const Dtype* in_p,const Dtype* in_a,
        const Dtype* in_g, const Dtype* in_l,const Dtype* in_nm ,Dtype* out){
        // Num:1    C:9     H:45    W:60
        // n:9*45*60
        //int Num = dim[0];
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];
        CUDA_KERNEL_LOOP(index,n){
            int num,c,h,w;
            int index_tmp = index;
            num = index_tmp / (C * H * W);
            index_tmp = index_tmp % (C * H * W);

            c = index_tmp / (H * W);
            index_tmp = index_tmp % (H * W);

            h = index_tmp / W;
            index_tmp = index_tmp % W;
            w = index_tmp;

            Dtype dIoG = in_iog[index];
            
            Dtype tx = in_p[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype ty = in_p[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype tw = in_p[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype th = in_p[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];
            tx = tx * in_nm[4] + in_nm[0];
            ty = ty * in_nm[5] + in_nm[1];
            tw = tw * in_nm[6] + in_nm[2];
            th = th * in_nm[7] + in_nm[3];

            Dtype px = in_a[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype py = in_a[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype pw = in_a[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype ph = in_a[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];

            Dtype Mx = in_g[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w];
            Dtype My = in_g[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w];
            Dtype Mw = in_g[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w];
            Dtype Mh = in_g[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w];

            Dtype Gx = px + tx * pw;
            Dtype Gy = py + ty * ph;
            Dtype Gw = pw * exp(tw);
            Dtype Gh = ph * exp(th);

            Dtype start_x = min(Gx, Mx);
            Dtype end_x = max(Gx + Gw, Mx + Mw);
            Dtype start_y = min(Gy, My);
            Dtype end_y = max(Gy + Gh, My + Mh);

            Dtype width_area = Gw + Mw - (end_x - start_x);
            Dtype height_area = Gh + Mh - (end_y - start_y);

            Dtype area2 = Mw * Mh;
            Dtype sym_c1 = (Gx) - (Mx);
            Dtype sym_c2 = (Gx + Gw) - (Mx + Mw);
            Dtype sym_k1 = (Gy) - (My);
            Dtype sym_k2 = (Gy + Gh) - (My + Mh);

            sym_c1 = (sym_c1 > 0) - (sym_c1 < 0);
            sym_c2 = (sym_c2 > 0) - (sym_c2 < 0);
            sym_k1 = (sym_k1 > 0) - (sym_k1 < 0);
            sym_k2 = (sym_k2 > 0) - (sym_k2 < 0);

            if(width_area <= 2 || height_area <=2 || area2 ==0){
                out[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w] = 0;
                out[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w] = 0;
                out[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w] = 0;
                out[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w] = 0;
            }
            else{
                out[(((num * C * 4 + 4 * c + 0) * H + h) * W) + w] = -dIoG * height_area  * pw * (sym_c1 + sym_c2) / (area2 * 2);
                out[(((num * C * 4 + 4 * c + 1) * H + h) * W) + w] = -dIoG * width_area * ph * (sym_k1 + sym_k2) / (area2 * 2);
                out[(((num * C * 4 + 4 * c + 2) * H + h) * W) + w] = dIoG * height_area * pw * exp(tw) * (1 - sym_c2) / (area2 * 2);
                out[(((num * C * 4 + 4 * c + 3) * H + h) * W) + w] = dIoG * width_area * ph * exp(th) * (1 - sym_k2) / (area2 * 2); 
            }
                
            
        }
    }

    /*
    template<typename Dtype>
    __global__ void ProIoUBackwardGPU(const int n,const int* dim, const Dtype* in_p,const Dtype* in_a,
        const Dtype* in_l, Dtype* out_tx, Dtype* out_ty, Dtype* out_tw, Dtype* out_th){
        // Num:1    C:9     H:45    W:60
        // n:24300 * 24300
        // define output: 24300*(24300 * 4)

        //note !!!!
         
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num1,c1,h1,w1;
            int num2,c2,h2,w2;
            int row,col;

            row = index / (C * H * W);
            col = index % (C * H * W); 

            int index1 = row;
            int index2 = col;
            
            num1 = index1 / (C * H * W);
            num2 = index2 / (C * H * W);
            index1 = index1 % (C * H * W);
            index2 = index2 % (C * H * W);

            c1 = index1 / (H * W);
            c2 = index2 / (H * W);
            index1 = index1 % (H * W);
            index2 = index2 % (H * W);

            h1 = index1 / W;
            h2 = index2 / W;
            index1 = index1 % W;
            index2 = index2 % W;
            w1 = index1;
            w2 = index2;

            int l1 = in_l[row];
            int l2 = in_l[col];

            if((col > row) && (l1 == 0 || l2 == 0 || l1 == l2)){
                out_tx[row * C * H * W + col] = 0;
                out_ty[row * C * H * W + col] = 0;
                out_tw[row * C * H * W + col] = 0;
                out_th[row * C * H * W + col] = 0;

                out_tx[col * C * H * W + row] = 0;
                out_ty[col * C * H * W + row] = 0;
                out_tw[col * C * H * W + row] = 0;
                out_th[col * C * H * W + row] = 0;
            }
            else if(col > row){
                Dtype tx1 = in_p[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype ty1 = in_p[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype tw1 = in_p[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype th1 = in_p[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype px1 = in_a[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype py1 = in_a[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype pw1 = in_a[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype ph1 = in_a[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype Gx1 = px1 + tx1 * pw1;
                Dtype Gy1 = py1 + ty1 * ph1;
                Dtype Gw1 = pw1 * exp(tw1);
                Dtype Gh1 = ph1 * exp(th1);

                Dtype tx2 = in_p[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype ty2 = in_p[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype tw2 = in_p[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype th2 = in_p[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype px2 = in_a[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype py2 = in_a[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype pw2 = in_a[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype ph2 = in_a[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype Gx2 = px2 + tx2 * pw2;
                Dtype Gy2 = py2 + ty2 * ph2;
                Dtype Gw2 = pw2 * exp(tw2);
                Dtype Gh2 = ph2 * exp(th2);

                Dtype start_x = min(Gx1 - 0.5 * Gw1, Gx2 - 0.5 * Gw2);
                Dtype end_x = max(Gx1 + 0.5 * Gw1, Gx2 + 0.5 * Gw2);
                Dtype start_y = min(Gy1 - 0.5 * Gh1, Gy2 - 0.5 * Gh2);
                Dtype end_y = max(Gy1 + 0.5 * Gh1, Gy2 + 0.5 * Gh2);

                Dtype width_area,height_area;
                width_area = Gw1 + Gw2 - (end_x - start_x);
                height_area = Gh1 + Gh2 - (end_y - start_y);
                Dtype area_all = Gw1*Gh1 + Gw2*Gh2;
                Dtype area_union = area_all - width_area * height_area;

                Dtype sym_c1 = (Gx1 - 0.5 * Gw1) - (Gx2 - 0.5 * Gw2);
                Dtype sym_c2 = (Gx1 + 0.5 * Gw1) - (Gx2 + 0.5 * Gw2);
                Dtype sym_k1 = (Gy1 - 0.5 * Gh1) - (Gy2 - 0.5 * Gh2);
                Dtype sym_k2 = (Gy1 + 0.5 * Gh1) - (Gy2 + 0.5 * Gh2);

                sym_c1 = (sym_c1 > 0) - (sym_c1 < 0);
                sym_c2 = (sym_c2 > 0) - (sym_c2 < 0);
                sym_k1 = (sym_k1 > 0) - (sym_k1 < 0);
                sym_k2 = (sym_k2 > 0) - (sym_k2 < 0); 

                Dtype dIoU_dw = height_area * area_all / (area_union*area_union);
                Dtype dIoU_dh = width_area * area_all / (area_union*area_union);
                Dtype dw_dx1 = -0.5 * (sym_c1 + sym_c2);
                Dtype dw_dx2 = -dw_dx1;
                Dtype dw_dw1 = 0.5 * (1 + 0.5*sym_c1 - 0.5*sym_c2);
                Dtype dw_dw2 = 0.5 * (1 - 0.5*sym_c1 + 0.5*sym_c2);
                Dtype dh_dy1 = -0.5 * (sym_k1 + sym_k2);
                Dtype dh_dy2 = -dh_dy1;
                Dtype dh_dh1 = 0.5 * (1 + 0.5*sym_k1 - 0.5*sym_k2);
                Dtype dh_dh2 = 0.5 * (1 - 0.5*sym_k1 + 0.5*sym_k2);
                
                Dtype dIoU_dw1 = Gh1*area_union - area_all * (Gh1 - dw_dw1 * height_area);
                Dtype dIoU_dw2 = Gh2*area_union - area_all * (Gh2 - dw_dw2 * height_area);
                Dtype dIoU_dh1 = Gw1*area_union - area_all * (Gw1 - dh_dh1 * width_area);
                Dtype dIoU_dh2 = Gw2*area_union - area_all * (Gw2 - dh_dh2 * width_area);
         
                Dtype dIoU_dtx1 = dIoU_dw * dw_dx1 * pw1;
                Dtype dIoU_dty1 = dIoU_dh * dh_dy1 * ph1;
                Dtype dIoU_dtw1 = dIoU_dw1 * pw1 * exp(tw1);
                Dtype dIoU_dth1 = dIoU_dh1 * ph1 * exp(th1);

                Dtype dIoU_dtx2 = dIoU_dw * dw_dx2 * pw2;
                Dtype dIoU_dty2 = dIoU_dh * dh_dy2 * ph2;
                Dtype dIoU_dtw2 = dIoU_dw2 * pw2 * exp(tw2);
                Dtype dIoU_dth2 = dIoU_dh2 * ph2 * exp(th2);

                out_tx[row * C * H * W + col] = dIoU_dtx1;
                out_ty[row * C * H * W + col] = dIoU_dty1;
                out_tw[row * C * H * W + col] = dIoU_dtw1;
                out_th[row * C * H * W + col] = dIoU_dth1;

                out_tx[col * C * H * W + row] = dIoU_dtx2;
                out_ty[col * C * H * W + row] = dIoU_dty2;
                out_tw[col * C * H * W + row] = dIoU_dtw2;
                out_th[col * C * H * W + row] = dIoU_dth2;
            }  
        }
    }*/

    /*template<typename Dtype>
    __global__ void ProIoUBackwardGPU(const int n, const int label, const int* dim, const Dtype* in_p,const Dtype* in_a,
        const Dtype* in_l, Dtype* out){
        // Num:1    C:9     H:45    W:60
        // n:24300 * 24300
        // define output: 24300*(24300 * 4)
        // label: 0 -> tx; 1 -> ty; 2 -> tw; 3 -> th

        //note !!!!
         
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num1,c1,h1,w1;
            int num2,c2,h2,w2;
            int row,col;

            row = index / (C * H * W);
            col = index % (C * H * W); 

            int index1 = row;
            int index2 = col;
            
            num1 = index1 / (C * H * W);
            num2 = index2 / (C * H * W);
            index1 = index1 % (C * H * W);
            index2 = index2 % (C * H * W);

            c1 = index1 / (H * W);
            c2 = index2 / (H * W);
            index1 = index1 % (H * W);
            index2 = index2 % (H * W);

            h1 = index1 / W;
            h2 = index2 / W;
            index1 = index1 % W;
            index2 = index2 % W;
            w1 = index1;
            w2 = index2;

            int l1 = in_l[row];
            int l2 = in_l[col];

            if((col > row) && (l1 == 0 || l2 == 0 || l1 == l2)){
                out[row * C * H * W + col] = 0;
                out[row * C * H * W + col] = 0;
            }
            else if(col > row){
                Dtype tx1 = in_p[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype ty1 = in_p[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype tw1 = in_p[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype th1 = in_p[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype px1 = in_a[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype py1 = in_a[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype pw1 = in_a[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype ph1 = in_a[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype Gx1 = px1 + tx1 * pw1;
                Dtype Gy1 = py1 + ty1 * ph1;
                Dtype Gw1 = pw1 * exp(tw1);
                Dtype Gh1 = ph1 * exp(th1);

                Dtype tx2 = in_p[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype ty2 = in_p[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype tw2 = in_p[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype th2 = in_p[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype px2 = in_a[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype py2 = in_a[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype pw2 = in_a[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype ph2 = in_a[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype Gx2 = px2 + tx2 * pw2;
                Dtype Gy2 = py2 + ty2 * ph2;
                Dtype Gw2 = pw2 * exp(tw2);
                Dtype Gh2 = ph2 * exp(th2);

                Dtype start_x = min(Gx1 - 0.5 * Gw1, Gx2 - 0.5 * Gw2);
                Dtype end_x = max(Gx1 + 0.5 * Gw1, Gx2 + 0.5 * Gw2);
                Dtype start_y = min(Gy1 - 0.5 * Gh1, Gy2 - 0.5 * Gh2);
                Dtype end_y = max(Gy1 + 0.5 * Gh1, Gy2 + 0.5 * Gh2);

                Dtype width_area,height_area;
                width_area = Gw1 + Gw2 - (end_x - start_x);
                height_area = Gh1 + Gh2 - (end_y - start_y);
                Dtype area_all = Gw1*Gh1 + Gw2*Gh2;
                Dtype area_union = area_all - width_area * height_area;

                Dtype sym_c1 = (Gx1 - 0.5 * Gw1) - (Gx2 - 0.5 * Gw2);
                Dtype sym_c2 = (Gx1 + 0.5 * Gw1) - (Gx2 + 0.5 * Gw2);
                Dtype sym_k1 = (Gy1 - 0.5 * Gh1) - (Gy2 - 0.5 * Gh2);
                Dtype sym_k2 = (Gy1 + 0.5 * Gh1) - (Gy2 + 0.5 * Gh2);

                sym_c1 = (sym_c1 > 0) - (sym_c1 < 0);
                sym_c2 = (sym_c2 > 0) - (sym_c2 < 0);
                sym_k1 = (sym_k1 > 0) - (sym_k1 < 0);
                sym_k2 = (sym_k2 > 0) - (sym_k2 < 0); 

                Dtype dIoU_dw = height_area * area_all / (area_union*area_union);
                Dtype dIoU_dh = width_area * area_all / (area_union*area_union);
                Dtype dw_dx1 = -0.5 * (sym_c1 + sym_c2);
                Dtype dw_dx2 = -dw_dx1;
                Dtype dw_dw1 = 0.5 * (1 + 0.5*sym_c1 - 0.5*sym_c2);
                Dtype dw_dw2 = 0.5 * (1 - 0.5*sym_c1 + 0.5*sym_c2);
                Dtype dh_dy1 = -0.5 * (sym_k1 + sym_k2);
                Dtype dh_dy2 = -dh_dy1;
                Dtype dh_dh1 = 0.5 * (1 + 0.5*sym_k1 - 0.5*sym_k2);
                Dtype dh_dh2 = 0.5 * (1 - 0.5*sym_k1 + 0.5*sym_k2);
                
                Dtype dIoU_dw1 = Gh1*area_union - area_all * (Gh1 - dw_dw1 * height_area);
                Dtype dIoU_dw2 = Gh2*area_union - area_all * (Gh2 - dw_dw2 * height_area);
                Dtype dIoU_dh1 = Gw1*area_union - area_all * (Gw1 - dh_dh1 * width_area);
                Dtype dIoU_dh2 = Gw2*area_union - area_all * (Gw2 - dh_dh2 * width_area);
         
                Dtype dIoU_dtx1 = dIoU_dw * dw_dx1 * pw1;
                Dtype dIoU_dty1 = dIoU_dh * dh_dy1 * ph1;
                Dtype dIoU_dtw1 = dIoU_dw1 * pw1 * exp(tw1);
                Dtype dIoU_dth1 = dIoU_dh1 * ph1 * exp(th1);

                Dtype dIoU_dtx2 = dIoU_dw * dw_dx2 * pw2;
                Dtype dIoU_dty2 = dIoU_dh * dh_dy2 * ph2;
                Dtype dIoU_dtw2 = dIoU_dw2 * pw2 * exp(tw2);
                Dtype dIoU_dth2 = dIoU_dh2 * ph2 * exp(th2);

                switch(label){
                    case 0:
                        out[row * C * H * W + col] = dIoU_dtx1;
                        out[col * C * H * W + row] = dIoU_dtx2;
                    case 1:
                        out[row * C * H * W + col] = dIoU_dty1;
                        out[col * C * H * W + row] = dIoU_dty2;
                    case 2:
                        out[row * C * H * W + col] = dIoU_dtw1;
                        out[col * C * H * W + row] = dIoU_dtw2;
                    case 3:
                        out[row * C * H * W + col] = dIoU_dth1;
                        out[col * C * H * W + row] = dIoU_dth2;

                }
            }
        }
    }*/

    template<typename Dtype>
    __global__ void ProIoUBackwardGPU(const int n, const int* dim, const Dtype* in_iou,const Dtype* in_p,const Dtype* in_a,
        const Dtype* in_l,const Dtype* in_nm ,Dtype* out){
        // Num:1    C:9     H:45    W:60
        // n:24300 * 24300
        // define output: 24300*(24300 * 4)
        // label: 0 -> tx; 1 -> ty; 2 -> tw; 3 -> th

        //note !!!!
         
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num1,c1,h1,w1;
            int num2,c2,h2,w2;
            int row,col;

            row = index / (C * H * W);
            col = index % (C * H * W); 

            int index1 = row;
            int index2 = col;
            
            num1 = index1 / (C * H * W);
            num2 = index2 / (C * H * W);
            index1 = index1 % (C * H * W);
            index2 = index2 % (C * H * W);

            c1 = index1 / (H * W);
            c2 = index2 / (H * W);
            index1 = index1 % (H * W);
            index2 = index2 % (H * W);

            h1 = index1 / W;
            h2 = index2 / W;
            index1 = index1 % W;
            index2 = index2 % W;
            w1 = index1;
            w2 = index2;

            int l1 = in_l[row];
            int l2 = in_l[col];

            Dtype dIoU = in_iou[index];

            if((col > row) && (l1 != 0 && l2 != 0 && l1 != l2)){
                Dtype tx1 = in_p[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype ty1 = in_p[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype tw1 = in_p[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype th1 = in_p[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
                tx1 = tx1 * in_nm[4] + in_nm[0];
                ty1 = ty1 * in_nm[5] + in_nm[1];
                tw1 = tw1 * in_nm[6] + in_nm[2];
                th1 = th1 * in_nm[7] + in_nm[3];
    
                Dtype px1 = in_a[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1];
                Dtype py1 = in_a[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1];
                Dtype pw1 = in_a[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1];
                Dtype ph1 = in_a[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1];
    
                Dtype Gx1 = px1 + tx1 * pw1;
                Dtype Gy1 = py1 + ty1 * ph1;
                Dtype Gw1 = pw1 * exp(tw1);
                Dtype Gh1 = ph1 * exp(th1);

                Dtype tx2 = in_p[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype ty2 = in_p[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype tw2 = in_p[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype th2 = in_p[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
                tx2 = tx2 * in_nm[4] + in_nm[0];
                ty2 = ty2 * in_nm[5] + in_nm[1];
                tw2 = tw2 * in_nm[6] + in_nm[2];
                th2 = th2 * in_nm[7] + in_nm[3];
    
                Dtype px2 = in_a[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2];
                Dtype py2 = in_a[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2];
                Dtype pw2 = in_a[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2];
                Dtype ph2 = in_a[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2];
    
                Dtype Gx2 = px2 + tx2 * pw2;
                Dtype Gy2 = py2 + ty2 * ph2;
                Dtype Gw2 = pw2 * exp(tw2);
                Dtype Gh2 = ph2 * exp(th2);

                Dtype start_x = min(Gx1, Gx2);
                Dtype end_x = max(Gx1 + Gw1, Gx2 + Gw2);
                Dtype start_y = min(Gy1, Gy2);
                Dtype end_y = max(Gy1 + Gh1, Gy2 + Gh2);

                

                Dtype width_area,height_area;
                width_area = Gw1 + Gw2 - (end_x - start_x);
                height_area = Gh1 + Gh2 - (end_y - start_y);
                Dtype area_all = Gw1*Gh1 + Gw2*Gh2;
                Dtype area_union = area_all - width_area * height_area;

                Dtype sym_c1 = (Gx1) - (Gx2);
                Dtype sym_c2 = (Gx1 + Gw1) - (Gx2 + Gw2);
                Dtype sym_k1 = (Gy1) - (Gy2);
                Dtype sym_k2 = (Gy1 + Gh1) - (Gy2 + Gh2);

                sym_c1 = (sym_c1 > 0) - (sym_c1 < 0);
                sym_c2 = (sym_c2 > 0) - (sym_c2 < 0);
                sym_k1 = (sym_k1 > 0) - (sym_k1 < 0);
                sym_k2 = (sym_k2 > 0) - (sym_k2 < 0);
                
                Dtype sign = 1;
                if(width_area <= 5 || height_area <= 5 ){
                    sign = 0;
                }

                Dtype dIoU_dw = height_area * area_all / (area_union*area_union);
                Dtype dIoU_dh = width_area * area_all / (area_union*area_union);
                Dtype dw_dx1 = -0.5 * (sym_c1 + sym_c2);
                Dtype dw_dx2 = -dw_dx1;
                Dtype dw_dw1 = 0.5 * (1 - sym_c2);
                Dtype dw_dw2 = 0.5 * (1 + sym_c2);
                Dtype dh_dy1 = -0.5 * (sym_k1 + sym_k2);
                Dtype dh_dy2 = -dh_dy1;
                Dtype dh_dh1 = 0.5 * (1 - sym_k2);
                Dtype dh_dh2 = 0.5 * (1 + sym_k2);
                
                Dtype dIoU_dw1 = (Gh1*area_union - area_all * (Gh1 - dw_dw1 * height_area)) / (area_union * area_union);
                Dtype dIoU_dw2 = (Gh2*area_union - area_all * (Gh2 - dw_dw2 * height_area)) / (area_union * area_union);
                Dtype dIoU_dh1 = (Gw1*area_union - area_all * (Gw1 - dh_dh1 * width_area)) / (area_union * area_union);
                Dtype dIoU_dh2 = (Gw2*area_union - area_all * (Gw2 - dh_dh2 * width_area)) / (area_union * area_union);
         
                Dtype dIoU_dtx1 = dIoU_dw * dw_dx1 * pw1;
                Dtype dIoU_dty1 = dIoU_dh * dh_dy1 * ph1;
                Dtype dIoU_dtw1 = dIoU_dw1 * pw1 * exp(tw1);
                Dtype dIoU_dth1 = dIoU_dh1 * ph1 * exp(th1);

                Dtype dIoU_dtx2 = dIoU_dw * dw_dx2 * pw2;
                Dtype dIoU_dty2 = dIoU_dh * dh_dy2 * ph2;
                Dtype dIoU_dtw2 = dIoU_dw2 * pw2 * exp(tw2);
                Dtype dIoU_dth2 = dIoU_dh2 * ph2 * exp(th2);

                caffe_gpu_atomic_add(sign * dIoU * dIoU_dtx1,&(out[(((num1 * C * 4 + 4 * c1 + 0) * H + h1) * W) + w1]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dty1,&(out[(((num1 * C * 4 + 4 * c1 + 1) * H + h1) * W) + w1]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dtw1,&(out[(((num1 * C * 4 + 4 * c1 + 2) * H + h1) * W) + w1]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dth1,&(out[(((num1 * C * 4 + 4 * c1 + 3) * H + h1) * W) + w1]));

                caffe_gpu_atomic_add(sign * dIoU * dIoU_dtx2,&(out[(((num2 * C * 4 + 4 * c2 + 0) * H + h2) * W) + w2]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dty2,&(out[(((num2 * C * 4 + 4 * c2 + 1) * H + h2) * W) + w2]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dtw2,&(out[(((num2 * C * 4 + 4 * c2 + 2) * H + h2) * W) + w2]));
                caffe_gpu_atomic_add(sign * dIoU * dIoU_dth2,&(out[(((num2 * C * 4 + 4 * c2 + 3) * H + h2) * W) + w2]));

                }
            }
    }

    template<typename Dtype>
    __global__ void SetZeroBackwardGPU(const int n, Dtype* out){
        CUDA_KERNEL_LOOP(index,n){
            out[index] = 0;
        }
    }

    /*
    template<typename Dtype>
    __global__ void SumIoUdiffBackwardGPU(const int n, const int* dim, const Dtype* in_tx, const Dtype* in_ty,
        const Dtype* in_tw, const Dtype* in_th, Dtype* out){
        // dim: 1 9 45 60
        // n : 24300 * 24300 
        int C = dim[1];
        int H = dim[2];
        int W = dim[3];

        CUDA_KERNEL_LOOP(index,n){
            int num1,c1,h1,w1;
            int num2,c2,h2,w2;
            int row = index / (C * H * W);
            int col = index % (C * H * W);

            int index1 = row;
            int index2 = col;
            
            num1 = index1 / (C * H * W);
            num2 = index2 / (C * H * W);
            index1 = index1 % (C * H * W);
            index2 = index2 % (C * H * W);

            c1 = index1 / (H * W);
            c2 = index2 / (H * W);
            index1 = index1 % (H * W);
            index2 = index2 % (H * W);

            h1 = index1 / W;
            h2 = index2 / W;
            index1 = index1 % W;
            index2 = index2 % W;
            w1 = index1;
            w2 = index2;
         
            caffe_gpu_atomic_add(in_tx[index2],&(out[((num1 * 4 * C + 4 * c1 + 0) * H +h1) * W + w1]));
            caffe_gpu_atomic_add(in_ty[index2],&(out[((num1 * 4 * C + 4 * c1 + 1) * H +h1) * W + w1]));
            caffe_gpu_atomic_add(in_tw[index2],&(out[((num1 * 4 * C + 4 * c1 + 2) * H +h1) * W + w1]));
            caffe_gpu_atomic_add(in_th[index2],&(out[((num1 * 4 * C + 4 * c1 + 3) * H +h1) * W + w1]));
            //atomicAdd(&(out_f[(((num * C * 4 + 4 * c + col % 4) * H + h)) * W + w]),in[index]);
            //out[(((num * C * 4 + 4 * c + col % 4) * H + h)) * W + w] += in[index];
        }
    }
    */
    

    
    

    template<typename Dtype>
    void RepulsionLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom){

        

        //debug diff min && max init
        Dtype min_Attr[4] = {999,999,999,999};
        Dtype max_Attr[4] = {0,0,0,0};
        Dtype min_RepGT[4] = {999,999,999,999};
        Dtype max_RepGT[4] = {0,0,0,0};
        Dtype min_RepBox[4] = {999,999,999,999};
        Dtype max_RepBox[4] = {0,0,0,0};

        // Attr derivative
        int count = bottom[0]->count();
        int num_anchor = count / 4;
        
        SmoothL1BackwardGPU<Dtype> <<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
                count,diff_Attr_.gpu_data(),diff_Attr_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;



        // RepGT derivative
        SmoothLnBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>> >(
                count, diff_RepGT_.gpu_data(), diff_RepGT_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;
        
        RepGTBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor,
            bottom[5]->gpu_shape(),diff_RepGT_.gpu_data(),bottom[0]->gpu_data(),bottom[3]->gpu_data(),
            bottom[4]->gpu_data(),bottom[5]->gpu_data(),bottom[7]->gpu_data(),diff_RepGT_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;

        


        // RepBox derivative
        SmoothLnBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num_anchor * num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor*num_anchor,
            iou_RepBox_.gpu_data(),iou_RepBox_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;

        SetZeroBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,diff_RepBox_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;

        ProIoUBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(num_anchor * num_anchor),CAFFE_CUDA_NUM_THREADS>>>(num_anchor * num_anchor,bottom[6]->gpu_shape(),
            iou_RepBox_.gpu_data(),bottom[0]->gpu_data(),bottom[3]->gpu_data(),bottom[6]->gpu_data(),bottom[7]->gpu_data(),diff_RepBox_.mutable_gpu_data());
        CUDA_POST_KERNEL_CHECK;
     
           

        // all together no weight
        Dtype coef_Attr = 1.0 / (diff_.height() * diff_.width() * bottom[0]->num());
        Dtype coef_RepGT = 0.5 /  (diff_.height() * diff_.width() * bottom[0]->num());

        Dtype nzeros_scal;
        if(nzeros_RepBox_.data_at(0,0,0,0) > 0){
            nzeros_scal = nzeros_RepBox_.data_at(0,0,0,0);
        }
        else{
            nzeros_scal = 1;
        }

        
        Dtype coef_RepBox = 0.5 / nzeros_scal;

        if(debug){
            std::cout<<"coef_Attr:"<<coef_Attr<<std::endl;
            std::cout<<"coef_RepGT:"<<coef_RepGT<<std::endl;
            std::cout<<"coef_RepBox:"<<coef_RepBox<<std::endl;
        }

        caffe_gpu_scal(count,coef_Attr, diff_Attr_.mutable_gpu_data());

        caffe_gpu_scal(count,coef_RepGT, diff_RepGT_.mutable_gpu_data());

        caffe_gpu_scal(count,coef_RepBox, diff_RepBox_.mutable_gpu_data());


        if(debug){
            
            const Dtype* p_diffAttr = diff_Attr_.cpu_data();
            for(int i=0; i<count; i++){
                int c = i / (60 * 45);
                c = c % 4;
                if(p_diffAttr[i] < min_Attr[c])
                    min_Attr[c] = p_diffAttr[i];
                if(p_diffAttr[i] > max_Attr[c])
                    max_Attr[c] = p_diffAttr[i];
            }
        }

        if(debug){
            
            const Dtype* p_diffRepGT = diff_RepGT_.cpu_data();
            for(int i=0; i<count; i++){
                int c = i / (60 * 45);
                c = c % 4;
                if(p_diffRepGT[i] < min_RepGT[c])
                    min_RepGT[c] = p_diffRepGT[i];
                if(p_diffRepGT[i] > max_RepGT[c])
                    max_RepGT[c] = p_diffRepGT[i];
            }
        }

        if(debug){
            
            const Dtype* p_diffRepBox = diff_RepBox_.cpu_data();
            for(int i=0; i<count; i++){
                int c = i / (60 * 45);
                c = c % 4;
                if(p_diffRepBox[i] < min_RepBox[c])
                    min_RepBox[c] = p_diffRepBox[i];
                if(p_diffRepBox[i] > max_RepBox[c])
                    max_RepBox[c] = p_diffRepBox[i];
            }
        }

        if(debug){
            std::cout<<"start check min && max diff"<<std::endl;
            for(int i=0; i<4; i++){
                std::cout<<"min diff_Attr_ is "<< i << " : " <<min_Attr[i]<<std::endl;
                std::cout<<"max diff_Attr_ is "<< i << " : " <<max_Attr[i]<<std::endl;
                std::cout<<"**********************************************"<<std::endl;
                std::cout<<"min diff_RepGT_ is "<< i << " : " <<min_RepGT[i]<<std::endl;
                std::cout<<"max diff_RepGT_ is "<< i << " : " <<max_RepGT[i]<<std::endl;
                std::cout<<"**********************************************"<<std::endl;
                std::cout<<"min diff_RepBox_ is "<< i << " : " <<min_RepBox[i]<<std::endl;
                std::cout<<"max diff_RepBox_ is "<< i << " : " <<max_RepBox[i]<<std::endl;
            }
            
        }


        if(debug){
            Dtype diff_Attr_sum;
            caffe_gpu_asum(count,diff_Attr_.gpu_data(),&diff_Attr_sum);
            std::cout<<"diff_Attr_sum is: "<<diff_Attr_sum<<std::endl;

            Dtype diff_RepGT_sum;
            caffe_gpu_asum(count,diff_RepGT_.gpu_data(),&diff_RepGT_sum);
            std::cout<<"diff_RepGT_sum is: "<<diff_RepGT_sum<<std::endl;

            Dtype diff_RepBox_sum;
            caffe_gpu_asum(count,diff_RepBox_.gpu_data(),&diff_RepBox_sum);
            std::cout<<"diff_RepBox_sum is: "<<diff_RepBox_sum<<std::endl;
        }


        caffe_gpu_add(count,diff_Attr_.gpu_data(), diff_RepGT_.gpu_data(),diff_.mutable_cpu_data());

        caffe_gpu_add(count,diff_RepBox_.gpu_data(),diff_.gpu_data(),diff_.mutable_gpu_data());

        

        
        // go on
        for(int i=0; i<2; i++){
            if(propagate_down[i]){
                const Dtype sign = (i==0) ? 1:-1;
                const Dtype alpha = sign * top[0]->cpu_diff()[0];
                
                caffe_gpu_axpby(bottom[i]->count(),alpha,diff_.gpu_data(),Dtype(0),bottom[i]->mutable_gpu_diff());
            }
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(RepulsionLossLayer);

} //namespace caffe