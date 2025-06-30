                                              
let tf=require("@tensorflow/tfjs");

function print(...datas){
 datas.forEach(data=>{
  data.print();
})
}


let {log}=console;
// Tensor Operation
let a=tf.fill([4,4],10);
let b=tf.fill([4,4],5);
let add=tf.add(a,b);
let subtraction=tf.sub(a,b);
let divition=tf.div(a,b);
let scalar=tf.scalar(5);
let multiplication=tf.mul(a,scalar);

//print(multiplication);
// matrix multiplication
const mul=tf.matMul(a,b);
//print(mul);

let tensor1d=tf.tensor1d([4,5,6,4]);
let mean=tf.mean(tensor1d);
let max=tf.max(tensor1d);
let sum=tf.sum(tensor1d)
//print(mean,max,sum)


// 1d to 2d
let reshape=tensor1d.reshape([2,2]);
//print(reshape)


// tensor to array
tensor1d.array().then(item=>{
//log(item)
})

// convert multi dymantion to 1d
a.data().then((item)=>{
//log(item)
})

async function array(t){
 let data=await t.array();
log(data)
}

//array(a)


const matrix=tf.tensor([
[1,1,0],
[3,0,1],
[0,1,1]
]);
//print(matrix)
//data type convert
let bool=matrix.cast('bool')
//print(bool)



// add all tensor
let c=tf.fill([4,4],1);
const stacked=tf.stack([a,b,c])
//stacked.print(true)


// copy matrix
let cloneMatrix=matrix.clone();
let transpose=cloneMatrix.transpose();
//print(transpose,cloneMatrix);


// Identity matrix
let eye=tf.eye(5,5);
//print(eye);




// Simple model  y=5x+2

let x=tf.tensor([[-2],[1],[0],[3],[5]]);
let y=tf.tensor([-8,7,2,17,27],[5,1])

async function train(value=0){
 const model=tf.sequential();
model.add(tf.layers.dense({
 units:1,
inputShape:[1]
}))

model.compile({
loss:'meanSquaredError',
optimizer:'sgd'
})

await model.fit(x,y,{
epochs:200
}).then(()=>{
console.log("Model train complite")
});


let predection=model.predict(tf.tensor([[value]]))

print(predection)

model.dispose();
}

// test
//train(2);
