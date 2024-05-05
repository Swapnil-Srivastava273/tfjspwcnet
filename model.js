"use strict";

const leakyReLU = tf.layers.leakyReLU({ alpha: 0.1 });
const heNorm = tf.initializers.heNormal(3);



const Conv2D = (filters, kernelSize, strides, name, dilationRate=1) => {
    const layer = tf.layers.conv2d({
        filters:filters,
        kernelSize:kernelSize,
        padding:'same',
        kernelInitializer:heNorm,
        dilationRate:dilationRate,
        strides:strides,
        // activation:leakyReLU,
        name:name
    });
    layer.name = name;
    layer.activation = leakyReLU;
    return layer;
};

const DeConv2D = (filters, kernelSize, strides, name) => {
    const layer = tf.layers.conv2dTranspose({
        filters:filters,
        kernelSize:kernelSize,
        strides:strides,
        padding:'same',
        name:name
    });
    layer.name = name;
    return layer;
};

const PredictFlow = (name) => {
    const layer = tf.layers.conv2d({
        filters:2,
        kernelSize:3,
        name:name,
        padding:'same'
    });
    layer.name = name;
    return layer;
};

function CostVolumn(c1, warp, search_range, name){
    const padded_lvl = tf.pad(warp, [[0,0], [search_range, search_range], [search_range, search_range], [0,0]]);
    const [_, h, w, _1] = c1.shape;
    const max_offset = search_range*2 + 1;

    let cost_vol = [];
    for(let y=0; y<max_offset; y++){
        for(let x=0; x<max_offset; x++){
            const slice = tf.slice(padded_lvl, [0,y,x,0], [-1,h,w,-1]);
            const cost = tf.mean(tf.mul(c1, slice), 3, true);
            cost_vol.push(cost);
        }
    }
    cost_vol = tf.concat(cost_vol, 3);
    const leaky = tf.layers.leakyReLU({ alpha: 0.1, name:name});
    leaky.name = name;
    cost_vol = leaky.apply(cost_vol);
    return cost_vol;
}

function bilinearWarp(x, flow){
    const [batch, h, w, filters] = x.shape;
    let [grid_y, grid_x] = tf.meshgrid(tf.range(0,h), tf.range(0,w), {'indexing':'ij'});
    // console.assert(batch == 1);
    // TODO: make this work for batch != 1
    grid_x = tf.reshape(grid_x, [batch, h, w]);
    grid_y = tf.reshape(grid_y, [batch, h, w]);
    let grid_b = tf.fill([batch,h,w], 0);
    grid_b = tf.cast(grid_b, "float32");
    grid_x = tf.cast(grid_x, "float32");
    grid_y = tf.cast(grid_y, "float32");

    const [fx, fy] = tf.unstack(flow, -1);
    const fx_0 = tf.floor(fx);
    const fx_1 = tf.add(fx_0,1);
    const fy_0 = tf.floor(fy);
    const fy_1 = tf.add(fy_0,1);

    const h_lim = tf.cast(h-1, "float32");
    const w_lim = tf.cast(w-1, "float32");

    const gy_0 = tf.clipByValue(tf.add(grid_y,fy_0), tf.scalar(0.0), h_lim);
    const gy_1 = tf.clipByValue(tf.add(grid_y,fy_1), tf.scalar(0.0), h_lim);
    const gx_0 = tf.clipByValue(tf.add(grid_x,fx_0), tf.scalar(0.0), w_lim);
    const gx_1 = tf.clipByValue(tf.add(grid_x,fy_1), tf.scalar(0.0), w_lim);

    const g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], 3), "int32");
    const g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], 3), "int32");
    const g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], 3), "int32");
    const g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], 3), "int32");

    const x_00 = tf.gatherND(x, g_00);
    const x_01 = tf.gatherND(x, g_01);
    const x_10 = tf.gatherND(x, g_10);
    const x_11 = tf.gatherND(x, g_11);

    const c_00 = tf.expandDims(tf.mul(tf.sub(fy_1, fy),tf.sub(fx_1,fx)), 3);
    const c_01 = tf.expandDims(tf.mul(tf.sub(fy_1, fy),tf.sub(fx,fx_0)), 3);
    const c_10 = tf.expandDims(tf.mul(tf.sub(fy, fy_0),tf.sub(fx_1,fx)), 3);
    const c_11 = tf.expandDims(tf.mul(tf.sub(fy, fy_0),tf.sub(fx,fx_0)), 3);

    return tf.add(tf.mul(c_00, x_00), tf.add(tf.mul(c_01,x_01), tf.add(tf.mul(c_10, x_10), tf.mul(c_11, x_11))));
    
}

// modeltf.weights['']

class PWCNet{
    constructor(){
        // super();
        this.conv1a  = Conv2D( 16, 3, 2, 'conv1a');
        this.conv1aa = Conv2D( 16, 3, 1, 'conv1aa');
        this.conv1b  = Conv2D( 16, 3, 1, 'conv1b');
        this.conv2a  = Conv2D( 32, 3, 2, 'conv2a');
        this.conv2aa = Conv2D( 32, 3, 1, 'conv2aa');
        this.conv2b  = Conv2D( 32, 3, 1, 'conv2b');
        this.conv3a  = Conv2D( 64, 3, 2, 'conv3a');
        this.conv3aa = Conv2D( 64, 3, 1, 'conv3aa');
        this.conv3b  = Conv2D( 64, 3, 1, 'conv3b');
        this.conv4a  = Conv2D( 96, 3, 2, 'conv4a');
        this.conv4aa = Conv2D( 96, 3, 1, 'conv4aa');
        this.conv4b  = Conv2D( 96, 3, 1, 'conv4b');
        this.conv5a  = Conv2D(128, 3, 2, 'conv5a');
        this.conv5aa = Conv2D(128, 3, 1, 'conv5aa');
        this.conv5b  = Conv2D(128, 3, 1, 'conv5b');
        this.conv6aa = Conv2D(196, 3, 2, 'conv6aa');
        this.conv6a  = Conv2D(196, 3, 1, 'conv6a');
        this.conv6b  = Conv2D(196, 3, 1, 'conv6b');

        this.LeakyReLU = leakyReLU;
        
        this.conv6_0 = Conv2D(128, 3, 1, 'conv6_0');
        this.conv6_1 = Conv2D(128, 3, 1, 'conv6_1');
        this.conv6_2 = Conv2D(96,  3, 1, 'conv6_2');
        this.conv6_3 = Conv2D(64,  3, 1, 'conv6_3');
        this.conv6_4 = Conv2D(32,  3, 1, 'conv6_4') ;    
        this.deconv_6 = DeConv2D(2, 4, 2, 'deconv_6') ;
        this.upfeat_6 = DeConv2D(2, 4, 2, 'upfeat_6') ;

        this.conv5_0 = Conv2D(128, 3, 1, 'conv5_0');
        this.conv5_1 = Conv2D(128, 3, 1, 'conv5_1');
        this.conv5_2 = Conv2D(96,  3, 1, 'conv5_2');
        this.conv5_3 = Conv2D(64,  3, 1, 'conv5_3');
        this.conv5_4 = Conv2D(32,  3, 1, 'conv5_4');
        this.deconv_5 = DeConv2D(2, 4, 2, 'deconv_5');
        this.upfeat_5 = DeConv2D(2, 4, 2, 'upfeat_5');

        this.conv4_0 = Conv2D(128, 3, 1, 'conv4_0');
        this.conv4_1 = Conv2D(128, 3, 1, 'conv4_1');
        this.conv4_2 = Conv2D(96,  3, 1, 'conv4_2');
        this.conv4_3 = Conv2D(64,  3, 1, 'conv4_3');
        this.conv4_4 = Conv2D(32,  3, 1, 'conv4_4');
        this.deconv_4 = DeConv2D(2, 4, 2, 'deconv_4'); 
        this.upfeat_4 = DeConv2D(2, 4, 2, 'upfeat_4') ;

        this.conv3_0 = Conv2D(128, 3, 1, 'conv3_0');
        this.conv3_1 = Conv2D(128, 3, 1, 'conv3_1');
        this.conv3_2 = Conv2D(96,  3, 1, 'conv3_2');
        this.conv3_3 = Conv2D(64,  3, 1, 'conv3_3');
        this.conv3_4 = Conv2D(32,  3, 1, 'conv3_4');
        this.deconv_3 = DeConv2D(2, 4, 2, 'deconv_3') ;
        this.upfeat_3 = DeConv2D(2, 4, 2, 'upfeat_3') ;

        this.conv2_0 = Conv2D(128, 3, 1, 'conv2_0');
        this.conv2_1 = Conv2D(128, 3, 1, 'conv2_1');
        this.conv2_2 = Conv2D(96,  3, 1, 'conv2_2');
        this.conv2_3 = Conv2D(64,  3, 1, 'conv2_3');
        this.conv2_4 = Conv2D(32,  3, 1, 'conv2_4');
        // this.deconv_2 = DeConv2D(2, 4, 2, 'deconv_2') ;

        this.dc_conv1 = Conv2D(128, 3, 1, "dc_conv1", 1);
        this.dc_conv2 = Conv2D(128, 3, 1, "dc_conv2", 2);
        this.dc_conv3 = Conv2D(128, 3, 1, "dc_conv3", 4);
        this.dc_conv4 = Conv2D(96,  3, 1, "dc_conv4", 8);
        this.dc_conv5 = Conv2D(64,  3, 1, "dc_conv5", 16);
        this.dc_conv6 = Conv2D(32,  3, 1, "dc_conv6", 1)

        this.flow6_out = PredictFlow('flow6_out');
        this.flow5_out = PredictFlow('flow5_out') ;
        this.flow4_out = PredictFlow('flow4_out') ;
        this.flow3_out = PredictFlow('flow3_out') ;
        this.flow2_out = PredictFlow('flow2_out') ;
        this.dc_conv7 = PredictFlow('dc_conv7');
    }
    assignWeights(model){
        this.call(tf.zeros([1,64,64,6]));
        const layer_names = ["conv1a","conv1aa","conv1b","conv2a","conv2aa","conv2b","conv3a","conv3aa","conv3b","conv4a","conv4aa","conv4b","conv5a","conv5aa","conv5b","conv6aa","conv6a","conv6b","conv6_0","conv6_1","conv6_2","conv6_3","conv6_4","deconv_6","upfeat_6","conv5_0","conv5_1","conv5_2","conv5_3","conv5_4","deconv_5","upfeat_5","conv4_0","conv4_1","conv4_2","conv4_3","conv4_4","deconv_4","upfeat_4","conv3_0","conv3_1","conv3_2","conv3_3","conv3_4","deconv_3","upfeat_3","conv2_0","conv2_1","conv2_2","conv2_3","conv2_4","dc_conv1","dc_conv2","dc_conv3","dc_conv4","dc_conv5","dc_conv6","flow6_out","flow5_out","flow4_out","flow3_out","flow2_out","dc_conv7"];
        const weight_names = [];
        for(const x in model.weights){
            if(x.includes("Bias") || x.includes("Conv2D") || x.includes("conv2d_transpose")) weight_names.push(x);
        }
        const layer_weights = {};
        for(const l of layer_names){
            layer_weights[l] = [null,null];
        }
        for(const l of layer_names){
            for(const w of weight_names){
                if(!w.includes(l+'/')) continue;
                if(w.includes("BiasAdd/ReadVariableOp")) layer_weights[l][1] = model.weights[w][0];
                else if(w.includes("Conv2D/ReadVariableOp")) layer_weights[l][0] = model.weights[w][0];
                else if(w.includes("conv2d_transpose/ReadVariableOp")) layer_weights[l][0] = model.weights[w][0];
            }
        }

        for(const l in layer_weights){
            try{
                this[l].setWeights(layer_weights[l]);
            }catch(e){
                debugger;
            }
        }
    }
    call(inputs, is_training=false){
        const im1 = tf.slice(inputs, [0,0,0,0], [-1,-1,-1,3]);
        const im2 = tf.slice(inputs,[0,0,0,3],[-1,-1,-1,-1]);
        const c11 = this.conv1b.apply(this.conv1aa.apply(this.conv1a.apply(im1)));
        const c21 = this.conv1b.apply(this.conv1aa.apply(this.conv1a.apply(im2)));
        const c12 = this.conv2b.apply(this.conv2aa.apply(this.conv2a.apply(c11)));
        const c22 = this.conv2b.apply(this.conv2aa.apply(this.conv2a.apply(c21)));
        const c13 = this.conv3b.apply(this.conv3aa.apply(this.conv3a.apply(c12)));
        const c23 = this.conv3b.apply(this.conv3aa.apply(this.conv3a.apply(c22)));
        const c14 = this.conv4b.apply(this.conv4aa.apply(this.conv4a.apply(c13)));
        const c24 = this.conv4b.apply(this.conv4aa.apply(this.conv4a.apply(c23)));
        const c15 = this.conv5b.apply(this.conv5aa.apply(this.conv5a.apply(c14)));
        const c25 = this.conv5b.apply(this.conv5aa.apply(this.conv5a.apply(c24)));
        const c16 = this.conv6b.apply(this.conv6a.apply(this.conv6aa.apply(c15)));
        const c26 = this.conv6b.apply(this.conv6a.apply(this.conv6aa.apply(c25)));

        // ### 6th flow    
        const corr6 = CostVolumn(c16, c26, 4);
        let x = tf.concat([this.conv6_0.apply(corr6), corr6], 3);
        x = tf.concat([this.conv6_1.apply(x), x], 3);
        x = tf.concat([this.conv6_2.apply(x), x], 3);
        x = tf.concat([this.conv6_3.apply(x), x], 3);
        x = tf.concat([this.conv6_4.apply(x), x], 3);
        
        const flow6 = this.flow6_out.apply(x);
        const up_flow6 = this.deconv_6.apply(flow6);
        const up_feat6 = this.upfeat_6.apply(x);

        // ### 5th flow
        const warp5 = bilinearWarp(c25, tf.mul(up_flow6,0.625));
        const corr5 = CostVolumn(c15, warp5, 4);

        x = tf.concat([corr5, c15, up_flow6, up_feat6], 3);
        x = tf.concat([this.conv5_0.apply(x), x], 3);
        x = tf.concat([this.conv5_1.apply(x), x], 3);
        x = tf.concat([this.conv5_2.apply(x), x], 3);
        x = tf.concat([this.conv5_3.apply(x), x], 3);
        x = tf.concat([this.conv5_4.apply(x), x], 3);
        const flow5 = this.flow5_out.apply(x);
        const up_flow5 = this.deconv_5.apply(flow5);
        const up_feat5 = this.upfeat_5.apply(x);

        // ### 4th flow
        const warp4 = bilinearWarp(c24, tf.mul(up_flow5,1.25));
        const corr4 = CostVolumn(c14, warp4, 4);

        x = tf.concat([corr4, c14, up_flow5, up_feat5], 3);
        x = tf.concat([this.conv4_0.apply(x), x], 3);
        x = tf.concat([this.conv4_1.apply(x), x], 3);
        x = tf.concat([this.conv4_2.apply(x), x], 3);
        x = tf.concat([this.conv4_3.apply(x), x], 3);
        x = tf.concat([this.conv4_4.apply(x), x], 3);
        const flow4 = this.flow4_out.apply(x);
        const up_flow4 = this.deconv_4.apply(flow4);
        const up_feat4 = this.upfeat_4.apply(x);

        // ### 3rd flow
        const warp3 = bilinearWarp(c23, tf.mul(up_flow4,2.5));
        const corr3 = CostVolumn(c13, warp3, 4);
        
        x = tf.concat([corr3, c13, up_flow4, up_feat4], 3);
        x = tf.concat([this.conv3_0.apply(x), x], 3);
        x = tf.concat([this.conv3_1.apply(x), x], 3);
        x = tf.concat([this.conv3_2.apply(x), x], 3);
        x = tf.concat([this.conv3_3.apply(x), x], 3);
        x = tf.concat([this.conv3_4.apply(x), x], 3);
        const flow3 = this.flow3_out.apply(x);
        const up_flow3 = this.deconv_3.apply(flow3);
        const up_feat3 = this.upfeat_3.apply(x);

        // # 2nd flow
        const warp2 = bilinearWarp(c22, tf.mul(up_flow3,5.0)); 
        const corr2 = CostVolumn(c12, warp2, 4);

        x = tf.concat([corr2, c12, up_flow3, up_feat3], 3);
        x = tf.concat([this.conv2_0.apply(x), x], 3);
        x = tf.concat([this.conv2_1.apply(x), x], 3);
        x = tf.concat([this.conv2_2.apply(x), x], 3);
        x = tf.concat([this.conv2_3.apply(x), x], 3);
        x = tf.concat([this.conv2_4.apply(x), x], 3);
        let flow2 = this.flow2_out.apply(x);

        x = this.dc_conv4.apply(this.dc_conv3.apply(this.dc_conv2.apply(this.dc_conv1.apply(x))));
        flow2 = tf.add(flow2,this.dc_conv7.apply(this.dc_conv6.apply(this.dc_conv5.apply(x))));

        if(is_training) return [flow6, flow5, flow4, flow3, flow2];
        else return flow2;
    }
}