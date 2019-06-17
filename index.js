// import 'boostrap/dist/css/bootstrap.css'
import * as tf from '@tensorflow/tfjs'
// import { activation } from '@tensorflow/tfjs-layers/dist/exports_layers';
import { MnistData } from './data'

var model;

function createLogEntry(entry) {
    document.getElementById('log').innerHTML += `<br>` + entry
}

function createModel(){
    createLogEntry(`Create Model ...`)
    model = tf.sequential()
    createLogEntry(`Model created`)

    createLogEntry(`Add layers`)
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPool2d({
        poolSize: [2,2],
        strides: [2,2]
    }));

    model.add(tf.layers.flatten())
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }))

    createLogEntry('Layers created')
    
    createLogEntry('Starting compiling....')
    model.compile({
        optimizer: tf.train.sgd(0.15),
        loss: 'cetegoricalCrossentropy'
    })
    createLogEntry('Compiled')
}

let data;
async function load() {
    createLogEntry('Loading MNIST　data...')
    data = new MnistData();
    await data.load();
    createLogEntry('Data loaded successfully')
}

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

async function train() {
    createLogEntry('start training...')
    for (let i = 0; i <BATCH_SIZE; i++){
        const batch = tf.tidy(()=>{
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch
        });

        await model.fit(
            batch.xs, batch.labels, {batchSize: BATCH_SIZE, epochs: 1}
        )
        tf.dispose(batch)
        await tf.nextFrame();
    }
    createLogEntry('Training complete')
}

async function main() {
    createModel();
    await load();
    await train();
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Ramdomly Select Test Data And Predict"
}

main();
