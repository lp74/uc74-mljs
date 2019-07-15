import { Matrix, inverse, pseudoInverse } from 'ml-matrix';

export default function linearRegression(xs, ys, lambda = 0){
    const X = new Matrix(xs.map(x => ([1].concat(x))));
    const Y = new Matrix(ys.map(y => ([y])));
    
    const m = X.rows; // samples
    const n = X.columns; // features

    const L = Matrix.eye(n, n).mul(lambda).set(0, 0, 0);
    
    try {
        return inverse(X.transpose().mmul(X).add(L)).mmul(X.transpose().mmul(Y))
    }
    catch(error){
        console.error(error);
    }
    
    try {
        return pseudoInverse(X.transpose().mmul(X).add(L)).mmul(X.transpose().mmul(Y))
    }
    catch(error){
        console.error(error);
    }
    return undefined;
};
