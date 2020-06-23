async function init() {
    model = await tf.loadModel('https://github.com/khluu/smartsheet/blob/master/tfjs/model.json');
    console.log('model loaded')
}