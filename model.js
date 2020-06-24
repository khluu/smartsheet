//import * as tf from '@tensorflow/tfjs';
//import * as ui from './ui.jss';

async function init() {
    model = await tf.loadModel('https://raw.githubusercontent.com/khluu/smartsheet/master/model.js');
    //console.log('model loaded')
}

async function predict() {
    init();
}