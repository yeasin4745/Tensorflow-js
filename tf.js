                                              
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



let d=tf.tensor([ [5,6],[4,7] ])
let diag=tf.diag(d);
//print(diag,d)
let linspace=tf.linspace(9,7,10);
//linspace.print();
//tf.ones([5,7]).print()
//tf.onesLike(d).print()
//tf.range(1,5,1).print()
let v=tf.randomNormal([2,2],5,3)
let n=tf.truncatedNormal([8,5],6,2); //Effician
//print(n)
let vari=tf.variable(v)
// re asine value
//vari.assign(n);
//print(vari)

let label=tf.tensor([2,1,5,4],[1,4],'int32')
//tf.oneHot(label,4).print()

//tf.zeros([5,4]).print()
//tf.zerosLike(n).print()



const A=tf.tensor([[4,2],[3,5]]);
let s1=tf.scalar(5)
let s2=tf.scalar(6)
let I=tf.eye(2,2)


let ans=tf.matMul(A,A).sub(A.mul(s1)).add(I.mul(s2))

// expect float dtype
let s=A.sigmoid()
//print(s)



// Equation solve
//6g-8f-c=25
//-2g+8f+c=-17
//8g+4f+c=-29

let D=tf.tensor([[6,-8,-1],[-2,8,1],[8,4,1]])
let Dg=tf.tensor([[25,-8,-1],[-17,8,1],[-20,4,1]])
let Df=tf.tensor([[6,25,-1],[-2,-17,1],[8,-20,1]])
let Dc=tf.tensor([[6,-8,25],[-2,8,-17],[8,4,-20]])


//print(D,Dg,Df,Dc)
//let det=tf.linalg.det(D);

// softmax function

var data=tf.variable(tf.tensor([2,1,1.5]))
//tf.softmax(data).print()


let buffer=D.bufferSync()

//console.log(buffer.get(1,2))
//log(buffer.values)

let buff=tf.buffer([2,2])
buff.set(55,0,0)
let bufff=buff.toTensor()
//print(bufff)

buff=tf.tensor([5,3,7]).buffer();
buff.then((i)=>{
//log(i)
})


  






// load csv file
async function loadData(){

let csvUrl=`https://raw.githubusercontent.com/yeasin4745/csv-datasets/refs/heads/main/data.csv`
let dataset=await tf.data.csv(csvUrl,{
 columnConfigs:{ Weight:{ isLabel:true}, Height:{}},
hasHeader: true,
configuredColumnsOnly:true

})

let batched=dataset.batch(2);
await batched.forEachAsync(batch=>{
 log(batch.xs ,'=>',batch.ys)
})



/*
await dataset.forEachAsync((row)=>{
log(row.xs, " ==>", row.ys);
})
*/


}

//loadData();


// Simple model  y=5x+2

//data set

let x=tf.truncatedNormal([30],0,10,'int32').reshape([30,1]);
let y=tf.mul(x,5).add(2);

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
