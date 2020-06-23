async function init() {
    model = await tf.loadModel('/tfjs/model.json');
    console.log('model loaded')
}