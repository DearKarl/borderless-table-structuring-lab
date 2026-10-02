"""Fail-closed interpretation of observed frozen Paddle preprocessing records."""
import math
def verify(response,width,height):
    try:
        if response['input_image_shape'][:2]!=[height,width]:return False,'returned original image shape differs'
        resize=[x for x in response['trace'] if x['stage']=='Resize']
        post=[x for x in response['trace'] if x['stage']=='actual_postprocess_inputs']
        if len(resize)!=1 or len(post)!=1:return False,'missing or ambiguous actual resize/postprocess trace'
        resized=resize[0]['after'][0];post_data=post[0]['before'][1][0]
        for d in [resized,post_data]:
            if d['ori_img_size']!=[width,height] or d['img_size']!=[800,800]:return False,'actual sizes differ from frozen resize contract'
            if len(d['scale_factors'])!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in d['scale_factors']):return False,'invalid actual scale values'
            if any(abs(a-b)>1e-7 for a,b in zip(d['scale_factors'],[800/width,800/height])):return False,'actual scale factors differ'
        if resized['img']['shape']!=[800,800,3]:return False,'resize output is not expected HWC image'
        if not any(x['stage']=='actual_runner_inputs_outputs' for x in response['trace']):return False,'missing actual runner trace'
        return True,'frozen native postprocessor received original size and measured resize/scale; use returned original-pixel boxes without inverse rescaling'
    except (KeyError,TypeError,IndexError):return False,'incomplete actual preprocessing evidence'
