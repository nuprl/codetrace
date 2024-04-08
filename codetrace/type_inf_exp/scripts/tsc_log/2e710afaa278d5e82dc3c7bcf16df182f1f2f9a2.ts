export function round(__typ0: __typ0, decimals: __typ0 = 0): __typ0 {
  const rounded =
    decimals === -1
      ? __typ0
      : Number(`${Math.round(Number(`${__typ0}e${decimals}`))}e-${decimals}`);
  return isNaN(rounded) ? 0 : rounded;
}

export function clamp(__typ0: __typ0, min: __typ0, max: __typ0) {
  return Math.min(Math.max(__typ0, min), max);
}

export type matrix = __typ0[][];

export function multiplyMatrices(
  a: matrix,
  b: matrix,
  flat = true
): matrix | __typ0[] {
  checkInput(a, b);
  const [matrixA, matrixB] =
    a.length <= b.length ? [a, toColumns(b)] : [b, toColumns(a)];

  const matrix = resultMatrix(a, b).map((_, i) =>
    _.map((_, j) => dotProduct(matrixA[i], matrixB[j]))
  );

  return flat ? matrix.flat() : matrix;
}

function checkInput(a: matrix, b: matrix) {
  const error =
    "Number of columns in the first matrix must equal number of rows in the second matrix.";
  if (a[0].length !== b.length) throw new Error(error);
}

function toColumns(matrix: matrix): matrix {
  return matrix[0].map((_, i) => matrix.map((x) => x[i]));
}

function resultMatrix(a: matrix, b: matrix): matrix {
  return new Array(Math.min(a.length, b.length)).fill(
    new Array(Math.min(a[0].length, b[0].length)).fill(0)
  );
}

function dotProduct(a: __typ0[], b: __typ0[]): __typ0 {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}