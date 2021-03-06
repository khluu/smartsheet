//import * as tf from '@tensorflow/tfjs';
//import * as ui from './ui.js';

async function init() {
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/khluu/smartsheet/master/tfjs/model.json', strict=false);
    console.log('model loaded');
    run();
}

function predict() {
    init();
}

function run() {
    console.log(model.getWeights()[0].print());

    var a = tf.tensor ([[[ 0.03709199  ,2.0334387,   0.18545994, -0.09037506 , 0.,
   -0.09366621 , 0.       ,   0.06876937 , 1.        ],
  [ 0.22255193 , 1.9430637  , 1.3353115  ,-1.8978761,  -0.15064421,
   -0.02889192 , 1.0254734  , 2.0753584  , 0.        ],
  [ 1.5578635  , 0.04518753 , 0.2596439  , 2.0786264 , -0.51372683,
   -0.83839387 , 0.75317705 , 0.83714885 , 0.        ],
  [ 1.8175074  , 2.1238139  , 1.2982196  ,-1.7171261  ,-0.7888955,
   -0.35728654 , 0.94608957,  0.7986207  , 0.        ],
  [ 3.115727   , 0.40668777 , 0.40801188 , 4.292815  , -0.08566584,
    0.2713467  , 0.41978744 , 1.0644654 ,  0.        ],
  [ 3.5237389  , 4.699503   ,-1.5207715 , -0.13556258,  0.6970405,
    0.98117626 , 0.6625193  , 0.8287222 ,  0.        ],
  [ 2.0029674  , 4.5639405  , 2.7818992 , -2.6660643  , 0.559263,
    0.1951211  , 1.4010531  , 1.3020507 ,  0.        ],
  [ 4.7848663 ,  1.8978761  ,-0.07418398 ,-0.13556258 ,-0.5007144,
   -0.18660872  ,0.08416176  ,0.13198519, -1.        ]]], dtype=tf.float32);
    console.log(a.shape);
    b = model.predict(a);
    //console.log(b.argMax().type);
    var c = b.dataSync();
    for(i = 0; i < c.length; i++) {
        console.log(c[i])
    }
    document.getElementById("demo").innerHTML = c.length;
}

document.getElementById("clickMe").onclick = predict;
document.getElementById("runMe").onclick = run;