# How to compute mean and sigma

```js
const n = YS.rows;
const T = sushiLR(xs, ys);
const P = XS.mul(T);
const E = YS.sub(P);
const u = Matrix.sum(E)/n;
const s = Math.sqrt(Matrix.sum(E.map(x => (x - u)).map(x => x*x))/n);
```