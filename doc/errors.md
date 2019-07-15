# How to compute Standard Error

```js
const n = YS.rows;
const T = sushiLR(xs, ys);
const P = XS.mul(T);
const E = YS.sub(P);
const s = Math.sqrt(Matrix.sum(E.map(x => x*x))/n);
```