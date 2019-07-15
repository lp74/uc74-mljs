require('milsushi/src/sushi');
const Matrix = Sushi.Matrix;
export default function linearRegression(xs, ys, lambda = 0){
    const X = Matrix.fromArray(xs.map(x => ([1].concat(x))));
    const Y = Matrix.fromArray(ys.map(y => ([y])));
    
    const m = X.rows; // samples
    const n = X.cols; // features

    const L = Matrix.eye(n).times(lambda).set(0, 0, 0);
    
    try {
        return X.t().mul(X).add(L).inverse().mul(X.t().mul(Y))
    }
    catch(error){
        console.error(error);
    }
    return undefined;
};
