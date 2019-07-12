import linearRegression from '../src/linear-regression'
import { Matrix, inverse, pseudoInverse } from 'ml-matrix';

describe('Vectorialized linear regression with regularization', () => {
    it('should compute linear regression without regularization', () => {
        const T = linearRegression([1, 2], [20, 30]);
        expect(T.get(0,0)).toBeCloseTo(10);
        expect(T.get(1,0)).toBeCloseTo(10);
    });
    it('should compute linear regression without regularization', () => {
        const xs = [1, 2, 3];
        const X = new Matrix([[1, 1], [1, 2], [1, 3]]);
        const T = linearRegression(xs, [10, 20, 30]);
        expect(X.mmul(T).to1DArray()).toEqual([10, 20, 30]);
    });
    it('should compute linear regression without regularization with a matrix as input', () => {
        const xs = [[1, 1], [2, 2]];
        const X = new Matrix(xs.map(x => ([1].concat(x))));
        const T = linearRegression(xs, [10, 20, 30]);
        expect(X.mmul(T).to1DArray().map(x => (+x.toFixed(0)))).toEqual([10, 20]);
    });
})