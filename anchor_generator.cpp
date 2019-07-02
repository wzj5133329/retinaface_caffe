#include "anchor_generator.h"
#include <iostream>
using namespace std;
AnchorGenerator::AnchorGenerator() {
}

AnchorGenerator::~AnchorGenerator() {
}

// init different anchors
int AnchorGenerator::Init(int stride, const AnchorCfg& cfg, bool dense_anchor) {
    CRect2f base_anchor(0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1);   //都是(0,0,15,15）)
    std::vector<CRect2f> ratio_anchors;
    // get ratio anchors
    _ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
    _scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

    anchor_stride = stride;

    anchor_num = preset_anchors.size();
    for (int i = 0; i < anchor_num; ++i) {
         cout<<"anchor_num :";
	       //preset_anchors[i].print();
    }
    return anchor_num;
}

int AnchorGenerator::FilterAnchor(const caffe::Blob<float> *cls, const caffe::Blob<float> *reg, \
	const caffe::Blob<float> *pts, std::vector<Anchor>& result,float confidence_threshold) {

    
    assert(cls->shape(1) == anchor_num*2);   //anchor_num=2
    assert(reg->shape(1) == anchor_num*4);
    int pts_length = 0;
    if (pts) {
      	assert(pts->shape(1) % anchor_num == 0);
      	pts_length = pts->shape(1)/anchor_num/2;             //pts_length=5 五个点
                    
    }

    int w = cls->shape(3);                  
    int h = cls->shape(2);
    int step = h*w;

    const float* clsData = cls->cpu_data();
    const float* regData = reg->cpu_data();
    const float* ptsDate = pts->cpu_data();

    for (int i = 0; i < w; ++i) {
	       for (int j = 0; j < h; ++j) 
         {
	             int id = j*w+i;
	             for (int a = 0; a < anchor_num; ++a) 
               {
            		if(clsData[(anchor_num + a)*step + id] > confidence_threshold) 
                {
            		    std::cout <<"clsData[(anchor_num + a)*step + id]: "<< clsData[(anchor_num + a)*step + id] << std::endl;
            		    CRect2f box(i * anchor_stride + preset_anchors[a][0],
            			    j * anchor_stride + preset_anchors[a][1],
            			    i * anchor_stride + preset_anchors[a][2],
            			    j * anchor_stride + preset_anchors[a][3]);

              		    //printf("box::%f %f %f %f\n", box[0], box[1], box[2], box[3]);
              		    CRect2f delta(regData[(a*4+0)*step+id],
              			    regData[(a*4+1)*step+id],
              			    regData[(a*4+2)*step+id],
              			    regData[(a*4+3)*step+id]);
              		    Anchor res;
              		    res.anchor = cv::Rect2f(box[0], box[1], box[2], box[3]);
              		    bbox_pred(box, delta, res.finalbox);
              		    res.score = clsData[(anchor_num + a)*step+id];
              		    res.center = cv::Point(i,j);
              		    //printf("center %d %d\n", i, j);
		    
        		          if (pts) {
                    		 std::vector<cv::Point2f> pts_delta(pts_length);
                    			for (int p = 0; p < pts_length; ++p) {
                              pts_delta[p].x = ptsDate[a*pts_length*2+p*2+id];
                              pts_delta[p].y = ptsDate[a*pts_length*2+p*2+1+id];
                    			    
                    			}                		
                			landmark_pred(box, pts_delta, res.pts);
                			
        		    }
		    
		                  result.push_back(res);
		          }
	     }
	}
    }

    return 0;
}

int AnchorGenerator::FilterAnchor(const caffe::Blob<float> *cls, const caffe::Blob<float> *reg, \
  const caffe::Blob<float> *pts, std::vector<Anchor>& result,float ratio_w,float ratio_h,float confidence_threshold) {

   
    ratiow = ratio_w;
    ratioh = ratio_h;
    assert(cls->shape(1) == anchor_num*2);   //anchor_num=2
    assert(reg->shape(1) == anchor_num*4);
    assert(pts->shape(1) == anchor_num*10);
    int pts_length = 0;
    if (pts) {
        assert(pts->shape(1) % anchor_num == 0);
        pts_length = pts->shape(1)/anchor_num/2;
        //cout <<"pts_length:"<<pts_length<<endl;            //pts_length=5 
    }

    int w = cls->shape(3);                  
    int h = cls->shape(2);
    int step = h*w;

    const float* clsData = cls->cpu_data();
    const float* regData = reg->cpu_data();
    const float* ptsDate = pts->cpu_data();

    for (int i = 0; i < w; ++i) {
         for (int j = 0; j < h; ++j) 
         {
               int id = j*w+i;
               for (int a = 0; a < anchor_num; ++a) 
               {
                if(clsData[(anchor_num + a)*step + id] > confidence_threshold) 
                {
                    std::cout <<"score: "<< clsData[(anchor_num + a)*step + id] << std::endl;
                    CRect2f box(i * anchor_stride + preset_anchors[a][0],
                      j * anchor_stride + preset_anchors[a][1],
                      i * anchor_stride + preset_anchors[a][2],
                      j * anchor_stride + preset_anchors[a][3]);

                      
                      CRect2f delta(regData[(a*4+0)*step+id],
                        regData[(a*4+1)*step+id],
                        regData[(a*4+2)*step+id],
                        regData[(a*4+3)*step+id]);
                      Anchor res;
                      res.anchor = cv::Rect2f(box[0], box[1], box[2], box[3]);
                      bbox_pred(box, delta, res.finalbox);
                      res.score = clsData[(anchor_num + a)*step+id];
                      res.center = cv::Point(i,j);
                      
        
                      if (pts) {                        
                         std::vector<cv::Point2f> pts_delta(pts_length);
                          for (int p = 0; p < pts_length; ++p) 
                          {
                              pts_delta[p].x = ptsDate[(a*10+p*2)*step+id];
                              pts_delta[p].y = ptsDate[(a*10+p*2+1)*step+id];

                              
                          }                     
                      landmark_pred(box, pts_delta, res.pts);
                      
                }
        
                      result.push_back(res);
              }
       }
  }
    }

    return 0;
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    ratio_anchors.clear();
    float sz = w * h;
    for (int s = 0; s < ratios.size(); ++s) {
      	float r = ratios[s];
      	float size_ratios = sz / r;
      	float ws = std::sqrt(size_ratios);
      	float hs = ws * r;
      	ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
		    y_ctr - 0.5 * (hs - 1),
		    x_ctr + 0.5 * (ws - 1),
		    y_ctr + 0.5 * (hs - 1)));
    }
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
    scale_anchors.clear();
    for (int a = 0; a < ratio_anchor.size(); ++a) {
	CRect2f anchor = ratio_anchor[a];
	float w = anchor[2] - anchor[0] + 1;
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

	for (int s = 0; s < scales.size(); ++s) {
	    float ws = w * scales[s];
	    float hs = h * scales[s];
	    scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
			y_ctr - 0.5 * (hs - 1),
			x_ctr + 0.5 * (ws - 1),
			y_ctr + 0.5 * (hs - 1)));
	}
    }
}


//得到最后的box  是左上角坐标与左下角坐标
void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect2f& box) {
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr;
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w;
    float pred_h = std::exp(dh) * h;

    box = cv::Rect2f((pred_ctr_x - 0.5 * (pred_w - 1.0))*ratiow,
	    (pred_ctr_y - 0.5 * (pred_h - 1.0))*ratioh,
	    (pred_ctr_x + 0.5 * (pred_w - 1.0))*ratiow,
	    (pred_ctr_y + 0.5 * (pred_h - 1.0))*ratioh);
}


//得到在原图中的landmark
//delta是对于设定锚点的偏移量
void AnchorGenerator::landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) {
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (int i = 0; i < delta.size(); ++i) {
        	pts[i].x = (delta[i].x*w + x_ctr)*ratiow;
        	pts[i].y = (delta[i].y*h + y_ctr)*ratioh;
    }
}


