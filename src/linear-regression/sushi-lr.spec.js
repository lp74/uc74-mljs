import sushiLR from './sushi-lr';
import LR from './linear-regression'

const Matrix = Sushi.Matrix;
const extractFromIndexes = indexes => (value, index) => { return (indexes.indexOf(index) > -1) }

describe('Vectorialized linear regression with regularization', () => {
    it('should compute linear regression without regularization', () => {
        const T = sushiLR([1, 2], [10, 20]);
        expect(T.get(0, 0)).toEqual(0);
        expect(T.get(1, 0)).toEqual(10);
    });
    it('Performance', () => {
        const N = 1e+3;
        const t0 = performance.now()
        for (var i = 0; i < N; i++) {
            const T = sushiLR([1, 2, 3], [10, 20, 33]);
        }

        const t1 = performance.now()
        for (var i = 0; i < N; i++) {
            const T = LR([1, 2, 3], [10, 20, 33]);
        }
        const t2 = performance.now();

        console.log(t1 - t0, t2 - t1);

        expect(t2 - t1).toBeGreaterThan(t1 - t0)
    })
    it('Sanityze', () => {

        const xs = [1, 2, 3, 4, 5, 6, 7];
        const ys = [10, undefined, NaN, null, 50, 'A', '40', true];

        const idx = ys.reduce((acc, curr, idx) => {
            return (typeof curr === 'number' && Number.isFinite(curr))
                ? acc.concat(idx)
                : acc;
        }, [])

        expect(idx).toEqual([0, 4])

        const _xs = xs.filter(extractFromIndexes(idx));
        const _ys = ys.filter(extractFromIndexes(idx))

        expect(_xs).toEqual([1, 5]);
        expect(_ys).toEqual([10, 50]);

        const T = sushiLR(_xs, _ys);
        expect(T.get(0, 0)).toEqual(0);
    })
    it('Standard Error', () => {
        const xs = [1, 2, 3, 4, 5];
        const ys = [1, 2, 1.30, 3.75, 2.25];

        const XS = Matrix.fromArray(xs.map(x => [1].concat(x)));
        const YS = Matrix.fromArray(ys.map(x => [x]));

        

        const T = sushiLR(xs, ys);
        expect(T.get(0, 0)).toBeCloseTo(0.785);
        expect(T.get(1, 0)).toBeCloseTo(0.425);

        const n = YS.rows;
        const P = XS.mul(T);
        const E = YS.sub(P);
        const s = Math.sqrt(Matrix.sum(E.map(x => x*x))/n);
        expect(s).toBeCloseTo(0.747);
    })
})

