/*
 * Copyright 2020, Ludovic Henry
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package dev.ludovic.blas;

public class JavaBLAS implements BLAS {

  private static final JavaBLAS instance = new JavaBLAS();

  //FIXME: remove dependency on F2jBLAS
  private static final com.github.fommil.netlib.F2jBLAS f2j = new com.github.fommil.netlib.F2jBLAS();

  protected JavaBLAS() {}

  public static BLAS getInstance() {
    return instance;
  }

  protected int loopBound(int index, int size) {
    return index - (index % size);
  }

  protected IllegalArgumentException illegalArgument(String method, int arg) {
    return new IllegalArgumentException(String.format("** On entry to '%s' parameter number %d had an illegal value", method, arg));
  }

  public double dasum(int n, double[] x, int incx) {
    return dasum(n, x, 0, incx);
  }

  public double dasum(int n, double[] x, int offsetx, int incx) {
    if (n <= 0) {
      return 0.0;
    }

    double sum = 0.0;
    for (int ix = incx < 0 ? (n - 1) * -incx : 0; incx < 0 ? ix >= 0 : ix < n * incx; ix += incx) {
      sum += Math.abs(x[offsetx + ix]);
    }
    return sum;
  }

  public float sasum(int n, float[] x, int incx) {
    return sasum(n, x, 0, incx);
  }

  public float sasum(int n, float[] x, int offsetx, int incx) {
    if (n <= 0) {
      return 0.0f;
    }

    float sum = 0.0f;
    for (int ix = incx < 0 ? (n - 1) * -incx : 0; incx < 0 ? ix >= 0 : ix < n * incx; ix += incx) {
      sum += Math.abs(x[offsetx + ix]);
    }
    return sum;
  }

  public void daxpy(int n, double alpha, double[] x, int incx, double[] y, int incy) {
    daxpy(n, alpha, x, 0, incx, y, 0, incy);
  }

  // y += alpha * x
  public void daxpy(int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }
    if (alpha == 0.0) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      y[offsety + iy] += alpha * x[offsetx + ix];
    }
  }

  public void saxpy(int n, float alpha, float[] x, int incx, float[] y, int incy) {
    saxpy(n, alpha, x, 0, incx, y, 0, incy);
  }

  // y += alpha * x
  public void saxpy(int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }
    if (alpha == 0.0f) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      y[offsety + iy] += alpha * x[offsetx + ix];
    }
  }

  public void dcopy(int n, double[] x, int incx, double[] y, int incy) {
    dcopy(n, x, 0, incx, y, 0, incy);
  }

  public void dcopy(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }

    if (x.length >= offsetx + n && incx == 1 && y.length >= offsety + n && incy == 1) {
      System.arraycopy(x, offsetx, y, offsety, n);
    } else {
      for (int ix = incx < 0 ? (n - 1) * -incx : 0,
               iy = incy < 0 ? (n - 1) * -incy : 0;
           (incx < 0 ? ix >= 0 : ix < n * incx)
            && (incy < 0 ? iy >= 0 : iy < n * incy);
           ix += incx, iy += incy) {
        y[offsety + iy] = x[offsetx + ix];
      }
    }
  }

  public void scopy(int n, float[] x, int incx, float[] y, int incy) {
    scopy(n, x, 0, incx, y, 0, incy);
  }

  public void scopy(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }

    if (x.length >= offsetx + n && incx == 1 && y.length >= offsety + n && incy == 1) {
      System.arraycopy(x, offsetx, y, offsety, n);
    } else {
      for (int ix = incx < 0 ? (n - 1) * -incx : 0,
               iy = incy < 0 ? (n - 1) * -incy : 0;
           (incx < 0 ? ix >= 0 : ix < n * incx)
            && (incy < 0 ? iy >= 0 : iy < n * incy);
           ix += incx, iy += incy) {
        y[offsety + iy] = x[offsetx + ix];
      }
    }
  }

  public double ddot(int n, double[] x, int incx, double[] y, int incy) {
    return ddot(n, x, 0, incx, y, 0, incy);
  }

  // sum(x * y)
  public double ddot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy) {
    if (n <= 0) {
      return 0.0;
    }

    double sum = 0.0;
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      sum += x[offsetx + ix] * y[offsety + iy];
    }
    return sum;
  }

  public float sdot(int n, float[] x, int incx, float[] y, int incy) {
    return sdot(n, x, 0, incx, y, 0, incy);
  }

  // sum(x * y)
  public float sdot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy) {
    if (n <= 0) {
      return 0.0f;
    }

    float sum = 0.0f;
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      sum += x[offsetx + ix] * y[offsety + iy];
    }
    return sum;
  }

  public float sdsdot(int n, float sb, float[] x, int incx, float[] y, int incy) {
    return sdsdot(n, sb, x, 0, incx, y, 0, incy);
  }

  public float sdsdot(int n, float sb, float[] x, int offsetx, int incx, float[] y, int offsety, int incy) {
    if (n <= 0) {
      return 0.0f;
    }

    double sum = sb;
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      sum += (double)(x[offsetx + ix]) * (double)(y[offsety + iy]);
    }
    return (float)sum;
  }

  public void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int lda, double[] x, int incx, double beta, double[] y, int incy) {
    dgbmv(trans, m, n, kl, ku, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void dgbmv(String trans, int m, int n, int kl, int ku, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    f2j.dgbmv(trans, m, n, kl, ku, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
  }

  public void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int lda, float[] x, int incx, float beta, float[] y, int incy) {
    sgbmv(trans, m, n, kl, ku, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void sgbmv(String trans, int m, int n, int kl, int ku, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    f2j.sgbmv(trans, m, n, kl, ku, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
  }

  public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int lda, double[] b, int ldb, double beta, double[] c, int ldc) {
    dgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  // c = alpha * a * b + beta * c
  public void dgemm(String transa, String transb, int m, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    if (!lsame("T", transa) && !lsame("N", transa) && !lsame("C", transa)) {
      throw illegalArgument("DGEMM", 1);
    }
    if (!lsame("T", transb) && !lsame("N", transb) && !lsame("C", transb)) {
      throw illegalArgument("DGEMM", 2);
    }
    if (m < 0) {
      throw illegalArgument("DGEMM", 3);
    }
    if (n < 0) {
      throw illegalArgument("DGEMM", 4);
    }
    if (k < 0) {
      throw illegalArgument("DGEMM", 5);
    }
    if (lda < Math.max(1, lsame("N", transa) ? m : k)) {
      throw illegalArgument("DGEMM", 8);
    }
    if (ldb < Math.max(1, lsame("N", transb) ? k : n)) {
      throw illegalArgument("DGEMM", 10);
    }
    if (ldc < Math.max(1, m)) {
      throw illegalArgument("DGEMM", 13);
    }
    if (m == 0 || n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0)) {
      return;
    }
    if (alpha == 0.0) {
      int col = 0;
      for (; col < loopBound(n, 4); col += 4) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
        }
      }
      for (; col < n; col += 1) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
        }
      }
    } else if (lsame("N", transa) && lsame("N", transb)) {
      dgemmNN(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if ((lsame("T", transa) || lsame("C", transa)) && lsame("N", transb)) {
      dgemmTN(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("N", transa) && (lsame("T", transb) || lsame("C", transb))) {
      dgemmNT(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if ((lsame("T", transa) || lsame("C", transa)) && (lsame("T", transb) || lsame("C", transb))) {
      dgemmTT(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    }
  }

  protected void dgemmNN(int m, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          dgemmNNKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void dgemmNNKernel(int cols, int cole, int rows, int rowe, int is, int ie, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    int col = cols;
    for (; col < loopBound(cole, 4); col += 4) {
      if (beta != 1.0) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
            c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
            c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
            c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0;
            c[offsetc + row + (col + 1) * ldc] = 0.0;
            c[offsetc + row + (col + 2) * ldc] = 0.0;
            c[offsetc + row + (col + 3) * ldc] = 0.0;
          }
        }
      }
      int i = is;
      for (; i < loopBound(ie, 4); i += 4) {
        double alphab00 = alpha * b[offsetb + (i + 0) + (col + 0) * ldb];
        double alphab01 = alpha * b[offsetb + (i + 1) + (col + 0) * ldb];
        double alphab02 = alpha * b[offsetb + (i + 2) + (col + 0) * ldb];
        double alphab03 = alpha * b[offsetb + (i + 3) + (col + 0) * ldb];
        double alphab10 = alpha * b[offsetb + (i + 0) + (col + 1) * ldb];
        double alphab11 = alpha * b[offsetb + (i + 1) + (col + 1) * ldb];
        double alphab12 = alpha * b[offsetb + (i + 2) + (col + 1) * ldb];
        double alphab13 = alpha * b[offsetb + (i + 3) + (col + 1) * ldb];
        double alphab20 = alpha * b[offsetb + (i + 0) + (col + 2) * ldb];
        double alphab21 = alpha * b[offsetb + (i + 1) + (col + 2) * ldb];
        double alphab22 = alpha * b[offsetb + (i + 2) + (col + 2) * ldb];
        double alphab23 = alpha * b[offsetb + (i + 3) + (col + 2) * ldb];
        double alphab30 = alpha * b[offsetb + (i + 0) + (col + 3) * ldb];
        double alphab31 = alpha * b[offsetb + (i + 1) + (col + 3) * ldb];
        double alphab32 = alpha * b[offsetb + (i + 2) + (col + 3) * ldb];
        double alphab33 = alpha * b[offsetb + (i + 3) + (col + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + (col + 0) * ldc] += alphab00 * a[offseta + row + (i + 0) * lda]
                                             +  alphab01 * a[offseta + row + (i + 1) * lda]
                                             +  alphab02 * a[offseta + row + (i + 2) * lda]
                                             +  alphab03 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 1) * ldc] += alphab10 * a[offseta + row + (i + 0) * lda]
                                             +  alphab11 * a[offseta + row + (i + 1) * lda]
                                             +  alphab12 * a[offseta + row + (i + 2) * lda]
                                             +  alphab13 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 2) * ldc] += alphab20 * a[offseta + row + (i + 0) * lda]
                                             +  alphab21 * a[offseta + row + (i + 1) * lda]
                                             +  alphab22 * a[offseta + row + (i + 2) * lda]
                                             +  alphab23 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 3) * ldc] += alphab30 * a[offseta + row + (i + 0) * lda]
                                             +  alphab31 * a[offseta + row + (i + 1) * lda]
                                             +  alphab32 * a[offseta + row + (i + 2) * lda]
                                             +  alphab33 * a[offseta + row + (i + 3) * lda];
        }
      }
      for (; i < ie; i += 1) {
        double alphab0 = alpha * b[offsetb + i + (col + 0) * ldb];
        double alphab1 = alpha * b[offsetb + i + (col + 1) * ldb];
        double alphab2 = alpha * b[offsetb + i + (col + 2) * ldb];
        double alphab3 = alpha * b[offsetb + i + (col + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + (col + 0) * ldc] += alphab0 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 1) * ldc] += alphab1 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 2) * ldc] += alphab2 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 3) * ldc] += alphab3 * a[offseta + row + i * lda];
        }
      }
    }
    for (; col < cole; col += 1) {
      if (beta != 1.0) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0;
          }
        }
      }
      int i = is;
      for (; i < loopBound(ie, 4); i += 4) {
        double alphab0 = alpha * b[offsetb + (i + 0) + col * ldb];
        double alphab1 = alpha * b[offsetb + (i + 1) + col * ldb];
        double alphab2 = alpha * b[offsetb + (i + 2) + col * ldb];
        double alphab3 = alpha * b[offsetb + (i + 3) + col * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab0 * a[offseta + row + (i + 0) * lda]
                                       +  alphab1 * a[offseta + row + (i + 1) * lda]
                                       +  alphab2 * a[offseta + row + (i + 2) * lda]
                                       +  alphab3 * a[offseta + row + (i + 3) * lda];
        }
      }
      for (; i < ie; i += 1) {
        double alphab = alpha * b[offsetb + i + col * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab * a[offseta + row + i * lda];
        }
      }
    }
  }

  protected void dgemmTN(int m, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          dgemmTNKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void dgemmTNKernel(int cols, int cole, int rows, int rowe, int is, int ie, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    int col = cols;
    for (; col < loopBound(cole, 4); col += 4) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        double sum00 = 0.0;
        double sum01 = 0.0;
        double sum02 = 0.0;
        double sum03 = 0.0;
        double sum10 = 0.0;
        double sum11 = 0.0;
        double sum12 = 0.0;
        double sum13 = 0.0;
        double sum20 = 0.0;
        double sum21 = 0.0;
        double sum22 = 0.0;
        double sum23 = 0.0;
        double sum30 = 0.0;
        double sum31 = 0.0;
        double sum32 = 0.0;
        double sum33 = 0.0;
        int i = is;
        for (; i < ie; i += 1) {
          sum00 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum01 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum02 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum03 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum10 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum11 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum12 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum13 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum20 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum21 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum22 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum23 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum30 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum31 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum32 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum33 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 3) * ldb];
        }
        if (beta != 0.0) {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (row + 0) + (col + 0) * ldc];
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01 + beta * c[offsetc + (row + 1) + (col + 0) * ldc];
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02 + beta * c[offsetc + (row + 2) + (col + 0) * ldc];
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03 + beta * c[offsetc + (row + 3) + (col + 0) * ldc];
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10 + beta * c[offsetc + (row + 0) + (col + 1) * ldc];
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (row + 1) + (col + 1) * ldc];
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12 + beta * c[offsetc + (row + 2) + (col + 1) * ldc];
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13 + beta * c[offsetc + (row + 3) + (col + 1) * ldc];
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20 + beta * c[offsetc + (row + 0) + (col + 2) * ldc];
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21 + beta * c[offsetc + (row + 1) + (col + 2) * ldc];
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (row + 2) + (col + 2) * ldc];
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23 + beta * c[offsetc + (row + 3) + (col + 2) * ldc];
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30 + beta * c[offsetc + (row + 0) + (col + 3) * ldc];
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31 + beta * c[offsetc + (row + 1) + (col + 3) * ldc];
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32 + beta * c[offsetc + (row + 2) + (col + 3) * ldc];
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (row + 3) + (col + 3) * ldc];
        } else {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00;
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01;
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02;
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03;
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10;
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11;
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12;
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13;
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20;
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21;
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22;
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23;
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30;
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31;
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32;
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33;
        }
      }
      for (; row < rowe; row += 1) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        int i = is;
        for (; i < ie; i += 1) {
          sum0 += a[offseta + i + row * lda] * b[offsetb + i + (col + 0) * ldb];
          sum1 += a[offseta + i + row * lda] * b[offsetb + i + (col + 1) * ldb];
          sum2 += a[offseta + i + row * lda] * b[offsetb + i + (col + 2) * ldb];
          sum3 += a[offseta + i + row * lda] * b[offsetb + i + (col + 3) * ldb];
        }
        if (beta != 0.0) {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0 + beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1 + beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2 + beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3 + beta * c[offsetc + row + (col + 3) * ldc];
        } else {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0;
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1;
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2;
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; col < cole; col += 1) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        int i = is;
        for (; i < ie; i += 1) {
          double bval = b[offsetb + i + col * ldb];
          sum0 += a[offseta + i + (row + 0) * ldb] * bval;
          sum1 += a[offseta + i + (row + 1) * ldb] * bval;
          sum2 += a[offseta + i + (row + 2) * ldb] * bval;
          sum3 += a[offseta + i + (row + 3) * ldb] * bval;
        }
        if (beta != 0.0) {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0 + beta * c[offsetc + (row + 0) + col * ldc];
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1 + beta * c[offsetc + (row + 1) + col * ldc];
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2 + beta * c[offsetc + (row + 2) + col * ldc];
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3 + beta * c[offsetc + (row + 3) + col * ldc];
        } else {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0;
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1;
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2;
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3;
        }
      }
      for (; row < rowe; row += 1) {
        double sum = 0.0;
        int i = is;
        for (; i < ie; i += 1) {
          sum += a[offseta + i + row * lda] * b[offsetb + i + col * ldb];
        }
        if (beta != 0.0) {
          c[offsetc + row + col * ldc] = alpha * sum + beta * c[offsetc + row + col * ldc];
        } else {
          c[offsetc + row + col * ldc] = alpha * sum;
        }
      }
    }
  }

  protected void dgemmNT(int m, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          dgemmNTKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void dgemmNTKernel(int cols, int cole, int rows, int rowe, int is, int ie, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    // C = beta * C
    if (beta != 1.0) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
            c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
            c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
            c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0;
            c[offsetc + row + (col + 1) * ldc] = 0.0;
            c[offsetc + row + (col + 2) * ldc] = 0.0;
            c[offsetc + row + (col + 3) * ldc] = 0.0;
          }
        }
      }
      for (; col < cole; col += 1) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0;
          }
        }
      }
    }
    // C += alpha * A * B
    int i = is;
    for (; i < loopBound(ie, 4); i += 4) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        double alphab00 = alpha * b[offsetb + (col + 0) + (i + 0) * ldb];
        double alphab01 = alpha * b[offsetb + (col + 1) + (i + 0) * ldb];
        double alphab02 = alpha * b[offsetb + (col + 2) + (i + 0) * ldb];
        double alphab03 = alpha * b[offsetb + (col + 3) + (i + 0) * ldb];
        double alphab10 = alpha * b[offsetb + (col + 0) + (i + 1) * ldb];
        double alphab11 = alpha * b[offsetb + (col + 1) + (i + 1) * ldb];
        double alphab12 = alpha * b[offsetb + (col + 2) + (i + 1) * ldb];
        double alphab13 = alpha * b[offsetb + (col + 3) + (i + 1) * ldb];
        double alphab20 = alpha * b[offsetb + (col + 0) + (i + 2) * ldb];
        double alphab21 = alpha * b[offsetb + (col + 1) + (i + 2) * ldb];
        double alphab22 = alpha * b[offsetb + (col + 2) + (i + 2) * ldb];
        double alphab23 = alpha * b[offsetb + (col + 3) + (i + 2) * ldb];
        double alphab30 = alpha * b[offsetb + (col + 0) + (i + 3) * ldb];
        double alphab31 = alpha * b[offsetb + (col + 1) + (i + 3) * ldb];
        double alphab32 = alpha * b[offsetb + (col + 2) + (i + 3) * ldb];
        double alphab33 = alpha * b[offsetb + (col + 3) + (i + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          double a0 = a[offseta + row + (i + 0) * lda];
          double a1 = a[offseta + row + (i + 1) * lda];
          double a2 = a[offseta + row + (i + 2) * lda];
          double a3 = a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 0) * ldc] += alphab00 * a0
                                             +  alphab10 * a1
                                             +  alphab20 * a2
                                             +  alphab30 * a3;
          c[offsetc + row + (col + 1) * ldc] += alphab01 * a0
                                             +  alphab11 * a1
                                             +  alphab21 * a2
                                             +  alphab31 * a3;
          c[offsetc + row + (col + 2) * ldc] += alphab02 * a0
                                             +  alphab12 * a1
                                             +  alphab22 * a2
                                             +  alphab32 * a3;
          c[offsetc + row + (col + 3) * ldc] += alphab03 * a0
                                             +  alphab13 * a1
                                             +  alphab23 * a2
                                             +  alphab33 * a3;
        }
      }
      for (; col < cole; col += 1) {
        double alphab0 = alpha * b[offsetb + col + (i + 0) * ldb];
        double alphab1 = alpha * b[offsetb + col + (i + 1) * ldb];
        double alphab2 = alpha * b[offsetb + col + (i + 2) * ldb];
        double alphab3 = alpha * b[offsetb + col + (i + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab0 * a[offseta + row + (i + 0) * lda]
                                       +  alphab1 * a[offseta + row + (i + 1) * lda]
                                       +  alphab2 * a[offseta + row + (i + 2) * lda]
                                       +  alphab3 * a[offseta + row + (i + 3) * lda];
        }
      }
    }
    for (; i < ie; i += 1) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        double alphab0 = alpha * b[offsetb + (col + 0) + i * ldb];
        double alphab1 = alpha * b[offsetb + (col + 1) + i * ldb];
        double alphab2 = alpha * b[offsetb + (col + 2) + i * ldb];
        double alphab3 = alpha * b[offsetb + (col + 3) + i * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          double aval = a[offseta + row + i * lda];
          c[offsetc + row + (col + 0) * ldc] += alphab0 * aval;
          c[offsetc + row + (col + 1) * ldc] += alphab1 * aval;
          c[offsetc + row + (col + 2) * ldc] += alphab2 * aval;
          c[offsetc + row + (col + 3) * ldc] += alphab3 * aval;
        }
      }
      for (; col < cole; col += 1) {
        double alphab = alpha * b[offsetb + col + i * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab * a[offseta + row + i * lda];
        }
      }
    }
  }

  protected void dgemmTT(int m, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    final int Tcol = 16, Trow = 16, Ti = 16;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          dgemmTTKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void dgemmTTKernel(int cols, int cole, int rows, int rowe, int is, int ie, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    int u = cols;
    for (; u < loopBound(cole, 4); u += 4) {
      int v = rows;
      for (; v < loopBound(rowe, 4); v += 4) {
        double sum00 = 0.0;
        double sum01 = 0.0;
        double sum02 = 0.0;
        double sum03 = 0.0;
        double sum10 = 0.0;
        double sum11 = 0.0;
        double sum12 = 0.0;
        double sum13 = 0.0;
        double sum20 = 0.0;
        double sum21 = 0.0;
        double sum22 = 0.0;
        double sum23 = 0.0;
        double sum30 = 0.0;
        double sum31 = 0.0;
        double sum32 = 0.0;
        double sum33 = 0.0;
        for (int w = is; w < ie; w++) {
          double b0 = b[offsetb + (u + 0) + w * ldb];
          double b1 = b[offsetb + (u + 1) + w * ldb];
          double b2 = b[offsetb + (u + 2) + w * ldb];
          double b3 = b[offsetb + (u + 3) + w * ldb];
          double a0 = a[offseta + w + (v + 0) * lda];
          double a1 = a[offseta + w + (v + 1) * lda];
          double a2 = a[offseta + w + (v + 2) * lda];
          double a3 = a[offseta + w + (v + 3) * lda];
          sum00 += a0 * b0;
          sum01 += a1 * b0;
          sum02 += a2 * b0;
          sum03 += a3 * b0;
          sum10 += a0 * b1;
          sum11 += a1 * b1;
          sum12 += a2 * b1;
          sum13 += a3 * b1;
          sum20 += a0 * b2;
          sum21 += a1 * b2;
          sum22 += a2 * b2;
          sum23 += a3 * b2;
          sum30 += a0 * b3;
          sum31 += a1 * b3;
          sum32 += a2 * b3;
          sum33 += a3 * b3;
        }
        if (beta != 0.0) {
          c[offsetc + (v + 0) + (u + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (v + 0) + (u + 0) * ldc];
          c[offsetc + (v + 1) + (u + 0) * ldc] = alpha * sum01 + beta * c[offsetc + (v + 1) + (u + 0) * ldc];
          c[offsetc + (v + 2) + (u + 0) * ldc] = alpha * sum02 + beta * c[offsetc + (v + 2) + (u + 0) * ldc];
          c[offsetc + (v + 3) + (u + 0) * ldc] = alpha * sum03 + beta * c[offsetc + (v + 3) + (u + 0) * ldc];
          c[offsetc + (v + 0) + (u + 1) * ldc] = alpha * sum10 + beta * c[offsetc + (v + 0) + (u + 1) * ldc];
          c[offsetc + (v + 1) + (u + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (v + 1) + (u + 1) * ldc];
          c[offsetc + (v + 2) + (u + 1) * ldc] = alpha * sum12 + beta * c[offsetc + (v + 2) + (u + 1) * ldc];
          c[offsetc + (v + 3) + (u + 1) * ldc] = alpha * sum13 + beta * c[offsetc + (v + 3) + (u + 1) * ldc];
          c[offsetc + (v + 0) + (u + 2) * ldc] = alpha * sum20 + beta * c[offsetc + (v + 0) + (u + 2) * ldc];
          c[offsetc + (v + 1) + (u + 2) * ldc] = alpha * sum21 + beta * c[offsetc + (v + 1) + (u + 2) * ldc];
          c[offsetc + (v + 2) + (u + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (v + 2) + (u + 2) * ldc];
          c[offsetc + (v + 3) + (u + 2) * ldc] = alpha * sum23 + beta * c[offsetc + (v + 3) + (u + 2) * ldc];
          c[offsetc + (v + 0) + (u + 3) * ldc] = alpha * sum30 + beta * c[offsetc + (v + 0) + (u + 3) * ldc];
          c[offsetc + (v + 1) + (u + 3) * ldc] = alpha * sum31 + beta * c[offsetc + (v + 1) + (u + 3) * ldc];
          c[offsetc + (v + 2) + (u + 3) * ldc] = alpha * sum32 + beta * c[offsetc + (v + 2) + (u + 3) * ldc];
          c[offsetc + (v + 3) + (u + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (v + 3) + (u + 3) * ldc];
        } else {
          c[offsetc + (v + 0) + (u + 0) * ldc] = alpha * sum00;
          c[offsetc + (v + 1) + (u + 0) * ldc] = alpha * sum01;
          c[offsetc + (v + 2) + (u + 0) * ldc] = alpha * sum02;
          c[offsetc + (v + 3) + (u + 0) * ldc] = alpha * sum03;
          c[offsetc + (v + 0) + (u + 1) * ldc] = alpha * sum10;
          c[offsetc + (v + 1) + (u + 1) * ldc] = alpha * sum11;
          c[offsetc + (v + 2) + (u + 1) * ldc] = alpha * sum12;
          c[offsetc + (v + 3) + (u + 1) * ldc] = alpha * sum13;
          c[offsetc + (v + 0) + (u + 2) * ldc] = alpha * sum20;
          c[offsetc + (v + 1) + (u + 2) * ldc] = alpha * sum21;
          c[offsetc + (v + 2) + (u + 2) * ldc] = alpha * sum22;
          c[offsetc + (v + 3) + (u + 2) * ldc] = alpha * sum23;
          c[offsetc + (v + 0) + (u + 3) * ldc] = alpha * sum30;
          c[offsetc + (v + 1) + (u + 3) * ldc] = alpha * sum31;
          c[offsetc + (v + 2) + (u + 3) * ldc] = alpha * sum32;
          c[offsetc + (v + 3) + (u + 3) * ldc] = alpha * sum33;
        }
      }
      for (; v < rowe; v += 1) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        for (int w = is; w < ie; w += 1) {
          double aval = a[offseta + w + v * lda];
          sum0 += aval * b[offsetb + (u + 0) + w * ldb];
          sum1 += aval * b[offsetb + (u + 1) + w * ldb];
          sum2 += aval * b[offsetb + (u + 2) + w * ldb];
          sum3 += aval * b[offsetb + (u + 3) + w * ldb];
        }
        if (beta != 0.0) {
          c[offsetc + v + (u + 0) * ldc] = alpha * sum0 + beta * c[offsetc + v + (u + 0) * ldc];
          c[offsetc + v + (u + 1) * ldc] = alpha * sum1 + beta * c[offsetc + v + (u + 1) * ldc];
          c[offsetc + v + (u + 2) * ldc] = alpha * sum2 + beta * c[offsetc + v + (u + 2) * ldc];
          c[offsetc + v + (u + 3) * ldc] = alpha * sum3 + beta * c[offsetc + v + (u + 3) * ldc];
        } else {
          c[offsetc + v + (u + 0) * ldc] = alpha * sum0;
          c[offsetc + v + (u + 1) * ldc] = alpha * sum1;
          c[offsetc + v + (u + 2) * ldc] = alpha * sum2;
          c[offsetc + v + (u + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; u < cole; u += 1) {
      int v = rows;
      for (; v < loopBound(rowe, 4); v += 4) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        for (int w = is; w < ie; w += 1) {
          double bval = b[offsetb + u + w * ldb];
          sum0 += a[offseta + w + (v + 0) * lda] * bval;
          sum1 += a[offseta + w + (v + 1) * lda] * bval;
          sum2 += a[offseta + w + (v + 2) * lda] * bval;
          sum3 += a[offseta + w + (v + 3) * lda] * bval;
        }
        if (beta != 0.0) {
          c[offsetc + (v + 0) + u * ldc] = alpha * sum0 + beta * c[offsetc + (v + 0) + u * ldc];
          c[offsetc + (v + 1) + u * ldc] = alpha * sum1 + beta * c[offsetc + (v + 1) + u * ldc];
          c[offsetc + (v + 2) + u * ldc] = alpha * sum2 + beta * c[offsetc + (v + 2) + u * ldc];
          c[offsetc + (v + 3) + u * ldc] = alpha * sum3 + beta * c[offsetc + (v + 3) + u * ldc];
        } else {
          c[offsetc + (v + 0) + u * ldc] = alpha * sum0;
          c[offsetc + (v + 1) + u * ldc] = alpha * sum1;
          c[offsetc + (v + 2) + u * ldc] = alpha * sum2;
          c[offsetc + (v + 3) + u * ldc] = alpha * sum3;
        }
      }
      for (; v < rowe; v += 1) {
        double sum = 0.0;
        for (int w = is; w < ie; w += 1) {
          sum += a[offseta + w + v * lda] * b[offsetb + u + w * ldb];
        }
        if (beta != 0.0) {
          c[offsetc + v + u * ldc] = alpha * sum + beta * c[offsetc + v + u * ldc];
        } else {
          c[offsetc + v + u * ldc] = alpha * sum;
        }
      }
    }
  }

  public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc) {
    sgemm(transa, transb, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  public void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    if (!lsame("T", transa) && !lsame("N", transa) && !lsame("C", transa)) {
      throw illegalArgument("SGEMM", 1);
    }
    if (!lsame("T", transb) && !lsame("N", transb) && !lsame("C", transb)) {
      throw illegalArgument("SGEMM", 2);
    }
    if (m < 0) {
      throw illegalArgument("SGEMM", 3);
    }
    if (n < 0) {
      throw illegalArgument("SGEMM", 4);
    }
    if (k < 0) {
      throw illegalArgument("SGEMM", 5);
    }
    if (lda < Math.max(1, lsame("N", transa) ? m : k)) {
      throw illegalArgument("SGEMM", 8);
    }
    if (ldb < Math.max(1, lsame("N", transb) ? k : n)) {
      throw illegalArgument("SGEMM", 10);
    }
    if (ldc < Math.max(1, m)) {
      throw illegalArgument("SGEMM", 13);
    }
    if (m == 0 || n == 0 || ((alpha == 0.0f || k == 0) && beta == 1.0f)) {
      return;
    }
    if (alpha == 0.0f) {
      int col = 0;
      for (; col < loopBound(n, 4); col += 4) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
        }
      }
      for (; col < n; col += 1) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
        }
      }
    } else if (lsame("N", transa) && lsame("N", transb)) {
      sgemmNN(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if ((lsame("T", transa) || lsame("C", transa)) && lsame("N", transb)) {
      sgemmTN(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("N", transa) && (lsame("T", transb) || lsame("C", transb))) {
      sgemmNT(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if ((lsame("T", transa) || lsame("C", transa)) && (lsame("T", transb) || lsame("C", transb))) {
      sgemmTT(m, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    }
  }

  protected void sgemmNN(int m, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          sgemmNNKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void sgemmNNKernel(int cols, int cole, int rows, int rowe, int is, int ie, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    int col = cols;
    for (; col < loopBound(cole, 4); col += 4) {
      if (beta != 1.0f) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0f) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
            c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
            c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
            c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0f;
            c[offsetc + row + (col + 1) * ldc] = 0.0f;
            c[offsetc + row + (col + 2) * ldc] = 0.0f;
            c[offsetc + row + (col + 3) * ldc] = 0.0f;
          }
        }
      }
      int i = is;
      for (; i < loopBound(ie, 4); i += 4) {
        float alphab00 = alpha * b[offsetb + (i + 0) + (col + 0) * ldb];
        float alphab01 = alpha * b[offsetb + (i + 1) + (col + 0) * ldb];
        float alphab02 = alpha * b[offsetb + (i + 2) + (col + 0) * ldb];
        float alphab03 = alpha * b[offsetb + (i + 3) + (col + 0) * ldb];
        float alphab10 = alpha * b[offsetb + (i + 0) + (col + 1) * ldb];
        float alphab11 = alpha * b[offsetb + (i + 1) + (col + 1) * ldb];
        float alphab12 = alpha * b[offsetb + (i + 2) + (col + 1) * ldb];
        float alphab13 = alpha * b[offsetb + (i + 3) + (col + 1) * ldb];
        float alphab20 = alpha * b[offsetb + (i + 0) + (col + 2) * ldb];
        float alphab21 = alpha * b[offsetb + (i + 1) + (col + 2) * ldb];
        float alphab22 = alpha * b[offsetb + (i + 2) + (col + 2) * ldb];
        float alphab23 = alpha * b[offsetb + (i + 3) + (col + 2) * ldb];
        float alphab30 = alpha * b[offsetb + (i + 0) + (col + 3) * ldb];
        float alphab31 = alpha * b[offsetb + (i + 1) + (col + 3) * ldb];
        float alphab32 = alpha * b[offsetb + (i + 2) + (col + 3) * ldb];
        float alphab33 = alpha * b[offsetb + (i + 3) + (col + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + (col + 0) * ldc] += alphab00 * a[offseta + row + (i + 0) * lda]
                                             +  alphab01 * a[offseta + row + (i + 1) * lda]
                                             +  alphab02 * a[offseta + row + (i + 2) * lda]
                                             +  alphab03 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 1) * ldc] += alphab10 * a[offseta + row + (i + 0) * lda]
                                             +  alphab11 * a[offseta + row + (i + 1) * lda]
                                             +  alphab12 * a[offseta + row + (i + 2) * lda]
                                             +  alphab13 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 2) * ldc] += alphab20 * a[offseta + row + (i + 0) * lda]
                                             +  alphab21 * a[offseta + row + (i + 1) * lda]
                                             +  alphab22 * a[offseta + row + (i + 2) * lda]
                                             +  alphab23 * a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 3) * ldc] += alphab30 * a[offseta + row + (i + 0) * lda]
                                             +  alphab31 * a[offseta + row + (i + 1) * lda]
                                             +  alphab32 * a[offseta + row + (i + 2) * lda]
                                             +  alphab33 * a[offseta + row + (i + 3) * lda];
        }
      }
      for (; i < ie; i += 1) {
        float alphab0 = alpha * b[offsetb + i + (col + 0) * ldb];
        float alphab1 = alpha * b[offsetb + i + (col + 1) * ldb];
        float alphab2 = alpha * b[offsetb + i + (col + 2) * ldb];
        float alphab3 = alpha * b[offsetb + i + (col + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + (col + 0) * ldc] += alphab0 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 1) * ldc] += alphab1 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 2) * ldc] += alphab2 * a[offseta + row + i * lda];
          c[offsetc + row + (col + 3) * ldc] += alphab3 * a[offseta + row + i * lda];
        }
      }
    }
    for (; col < cole; col += 1) {
      if (beta != 1.0f) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0f) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0f;
          }
        }
      }
      int i = is;
      for (; i < loopBound(ie, 4); i += 4) {
        float alphab0 = alpha * b[offsetb + (i + 0) + col * ldb];
        float alphab1 = alpha * b[offsetb + (i + 1) + col * ldb];
        float alphab2 = alpha * b[offsetb + (i + 2) + col * ldb];
        float alphab3 = alpha * b[offsetb + (i + 3) + col * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab0 * a[offseta + row + (i + 0) * lda]
                                       +  alphab1 * a[offseta + row + (i + 1) * lda]
                                       +  alphab2 * a[offseta + row + (i + 2) * lda]
                                       +  alphab3 * a[offseta + row + (i + 3) * lda];
        }
      }
      for (; i < ie; i += 1) {
        float alphab = alpha * b[offsetb + i + col * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab * a[offseta + row + i * lda];
        }
      }
    }
  }

  protected void sgemmTN(int m, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          sgemmTNKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void sgemmTNKernel(int cols, int cole, int rows, int rowe, int is, int ie, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    int col = cols;
    for (; col < loopBound(cole, 4); col += 4) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        float sum00 = 0.0f;
        float sum01 = 0.0f;
        float sum02 = 0.0f;
        float sum03 = 0.0f;
        float sum10 = 0.0f;
        float sum11 = 0.0f;
        float sum12 = 0.0f;
        float sum13 = 0.0f;
        float sum20 = 0.0f;
        float sum21 = 0.0f;
        float sum22 = 0.0f;
        float sum23 = 0.0f;
        float sum30 = 0.0f;
        float sum31 = 0.0f;
        float sum32 = 0.0f;
        float sum33 = 0.0f;
        int i = is;
        for (; i < ie; i += 1) {
          sum00 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum01 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum02 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum03 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 0) * ldb];
          sum10 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum11 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum12 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum13 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 1) * ldb];
          sum20 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum21 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum22 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum23 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 2) * ldb];
          sum30 += a[offseta + i + (row + 0) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum31 += a[offseta + i + (row + 1) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum32 += a[offseta + i + (row + 2) * lda] * b[offsetb + i + (col + 3) * ldb];
          sum33 += a[offseta + i + (row + 3) * lda] * b[offsetb + i + (col + 3) * ldb];
        }
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (row + 0) + (col + 0) * ldc];
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01 + beta * c[offsetc + (row + 1) + (col + 0) * ldc];
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02 + beta * c[offsetc + (row + 2) + (col + 0) * ldc];
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03 + beta * c[offsetc + (row + 3) + (col + 0) * ldc];
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10 + beta * c[offsetc + (row + 0) + (col + 1) * ldc];
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (row + 1) + (col + 1) * ldc];
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12 + beta * c[offsetc + (row + 2) + (col + 1) * ldc];
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13 + beta * c[offsetc + (row + 3) + (col + 1) * ldc];
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20 + beta * c[offsetc + (row + 0) + (col + 2) * ldc];
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21 + beta * c[offsetc + (row + 1) + (col + 2) * ldc];
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (row + 2) + (col + 2) * ldc];
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23 + beta * c[offsetc + (row + 3) + (col + 2) * ldc];
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30 + beta * c[offsetc + (row + 0) + (col + 3) * ldc];
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31 + beta * c[offsetc + (row + 1) + (col + 3) * ldc];
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32 + beta * c[offsetc + (row + 2) + (col + 3) * ldc];
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (row + 3) + (col + 3) * ldc];
        } else {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00;
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01;
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02;
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03;
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10;
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11;
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12;
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13;
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20;
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21;
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22;
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23;
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30;
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31;
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32;
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33;
        }
      }
      for (; row < rowe; row += 1) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = is;
        for (; i < ie; i += 1) {
          sum0 += a[offseta + i + row * lda] * b[offsetb + i + (col + 0) * ldb];
          sum1 += a[offseta + i + row * lda] * b[offsetb + i + (col + 1) * ldb];
          sum2 += a[offseta + i + row * lda] * b[offsetb + i + (col + 2) * ldb];
          sum3 += a[offseta + i + row * lda] * b[offsetb + i + (col + 3) * ldb];
        }
        if (beta != 0.0f) {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0 + beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1 + beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2 + beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3 + beta * c[offsetc + row + (col + 3) * ldc];
        } else {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0;
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1;
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2;
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; col < cole; col += 1) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        int i = is;
        for (; i < ie; i += 1) {
          float bval = b[offsetb + i + col * ldb];
          sum0 += a[offseta + i + (row + 0) * ldb] * bval;
          sum1 += a[offseta + i + (row + 1) * ldb] * bval;
          sum2 += a[offseta + i + (row + 2) * ldb] * bval;
          sum3 += a[offseta + i + (row + 3) * ldb] * bval;
        }
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0 + beta * c[offsetc + (row + 0) + col * ldc];
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1 + beta * c[offsetc + (row + 1) + col * ldc];
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2 + beta * c[offsetc + (row + 2) + col * ldc];
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3 + beta * c[offsetc + (row + 3) + col * ldc];
        } else {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0;
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1;
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2;
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3;
        }
      }
      for (; row < rowe; row += 1) {
        float sum = 0.0f;
        int i = is;
        for (; i < ie; i += 1) {
          sum += a[offseta + i + row * lda] * b[offsetb + i + col * ldb];
        }
        if (beta != 0.0f) {
          c[offsetc + row + col * ldc] = alpha * sum + beta * c[offsetc + row + col * ldc];
        } else {
          c[offsetc + row + col * ldc] = alpha * sum;
        }
      }
    }
  }

  protected void sgemmNT(int m, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    final int Tcol = n, Trow = m, Ti = k;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          sgemmNTKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void sgemmNTKernel(int cols, int cole, int rows, int rowe, int is, int ie, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    // C = beta * C
    if (beta != 1.0f) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0f) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
            c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
            c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
            c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0f;
            c[offsetc + row + (col + 1) * ldc] = 0.0f;
            c[offsetc + row + (col + 2) * ldc] = 0.0f;
            c[offsetc + row + (col + 3) * ldc] = 0.0f;
          }
        }
      }
      for (; col < cole; col += 1) {
        int row = rows;
        for (; row < rowe; row += 1) {
          if (beta != 0.0f) {
            c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          } else {
            c[offsetc + row + (col + 0) * ldc] = 0.0f;
          }
        }
      }
    }
    // C += alpha * A * B
    int i = is;
    for (; i < loopBound(ie, 4); i += 4) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        float alphab00 = alpha * b[offsetb + (col + 0) + (i + 0) * ldb];
        float alphab01 = alpha * b[offsetb + (col + 1) + (i + 0) * ldb];
        float alphab02 = alpha * b[offsetb + (col + 2) + (i + 0) * ldb];
        float alphab03 = alpha * b[offsetb + (col + 3) + (i + 0) * ldb];
        float alphab10 = alpha * b[offsetb + (col + 0) + (i + 1) * ldb];
        float alphab11 = alpha * b[offsetb + (col + 1) + (i + 1) * ldb];
        float alphab12 = alpha * b[offsetb + (col + 2) + (i + 1) * ldb];
        float alphab13 = alpha * b[offsetb + (col + 3) + (i + 1) * ldb];
        float alphab20 = alpha * b[offsetb + (col + 0) + (i + 2) * ldb];
        float alphab21 = alpha * b[offsetb + (col + 1) + (i + 2) * ldb];
        float alphab22 = alpha * b[offsetb + (col + 2) + (i + 2) * ldb];
        float alphab23 = alpha * b[offsetb + (col + 3) + (i + 2) * ldb];
        float alphab30 = alpha * b[offsetb + (col + 0) + (i + 3) * ldb];
        float alphab31 = alpha * b[offsetb + (col + 1) + (i + 3) * ldb];
        float alphab32 = alpha * b[offsetb + (col + 2) + (i + 3) * ldb];
        float alphab33 = alpha * b[offsetb + (col + 3) + (i + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          float a0 = a[offseta + row + (i + 0) * lda];
          float a1 = a[offseta + row + (i + 1) * lda];
          float a2 = a[offseta + row + (i + 2) * lda];
          float a3 = a[offseta + row + (i + 3) * lda];
          c[offsetc + row + (col + 0) * ldc] += alphab00 * a0
                                             +  alphab10 * a1
                                             +  alphab20 * a2
                                             +  alphab30 * a3;
          c[offsetc + row + (col + 1) * ldc] += alphab01 * a0
                                             +  alphab11 * a1
                                             +  alphab21 * a2
                                             +  alphab31 * a3;
          c[offsetc + row + (col + 2) * ldc] += alphab02 * a0
                                             +  alphab12 * a1
                                             +  alphab22 * a2
                                             +  alphab32 * a3;
          c[offsetc + row + (col + 3) * ldc] += alphab03 * a0
                                             +  alphab13 * a1
                                             +  alphab23 * a2
                                             +  alphab33 * a3;
        }
      }
      for (; col < cole; col += 1) {
        float alphab0 = alpha * b[offsetb + col + (i + 0) * ldb];
        float alphab1 = alpha * b[offsetb + col + (i + 1) * ldb];
        float alphab2 = alpha * b[offsetb + col + (i + 2) * ldb];
        float alphab3 = alpha * b[offsetb + col + (i + 3) * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab0 * a[offseta + row + (i + 0) * lda]
                                       +  alphab1 * a[offseta + row + (i + 1) * lda]
                                       +  alphab2 * a[offseta + row + (i + 2) * lda]
                                       +  alphab3 * a[offseta + row + (i + 3) * lda];
        }
      }
    }
    for (; i < ie; i += 1) {
      int col = 0;
      for (; col < loopBound(cole, 4); col += 4) {
        float alphab0 = alpha * b[offsetb + (col + 0) + i * ldb];
        float alphab1 = alpha * b[offsetb + (col + 1) + i * ldb];
        float alphab2 = alpha * b[offsetb + (col + 2) + i * ldb];
        float alphab3 = alpha * b[offsetb + (col + 3) + i * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          float aval = a[offseta + row + i * lda];
          c[offsetc + row + (col + 0) * ldc] += alphab0 * aval;
          c[offsetc + row + (col + 1) * ldc] += alphab1 * aval;
          c[offsetc + row + (col + 2) * ldc] += alphab2 * aval;
          c[offsetc + row + (col + 3) * ldc] += alphab3 * aval;
        }
      }
      for (; col < cole; col += 1) {
        float alphab = alpha * b[offsetb + col + i * ldb];
        int row = rows;
        for (; row < rowe; row += 1) {
          c[offsetc + row + col * ldc] += alphab * a[offseta + row + i * lda];
        }
      }
    }
  }

  protected void sgemmTT(int m, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    final int Tcol = 16, Trow = 16, Ti = 16;
    for (int col = 0; col < n; col += Tcol) {
      for (int row = 0; row < m; row += Trow) {
        for (int i = 0; i < k; i += Ti) {
          sgemmTTKernel(col, Math.min(col + Tcol, n), row, Math.min(row + Trow, m), i, Math.min(i + Ti, k),
                        alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
      }
    }
  }

  protected void sgemmTTKernel(int cols, int cole, int rows, int rowe, int is, int ie, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    int col = cols;
    for (; col < loopBound(cole, 4); col += 4) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        float sum00 = 0.0f;
        float sum01 = 0.0f;
        float sum02 = 0.0f;
        float sum03 = 0.0f;
        float sum10 = 0.0f;
        float sum11 = 0.0f;
        float sum12 = 0.0f;
        float sum13 = 0.0f;
        float sum20 = 0.0f;
        float sum21 = 0.0f;
        float sum22 = 0.0f;
        float sum23 = 0.0f;
        float sum30 = 0.0f;
        float sum31 = 0.0f;
        float sum32 = 0.0f;
        float sum33 = 0.0f;
        for (int i = is; i < ie; i++) {
          float b0 = b[offsetb + (col + 0) + i * ldb];
          float b1 = b[offsetb + (col + 1) + i * ldb];
          float b2 = b[offsetb + (col + 2) + i * ldb];
          float b3 = b[offsetb + (col + 3) + i * ldb];
          float a0 = a[offseta + i + (row + 0) * lda];
          float a1 = a[offseta + i + (row + 1) * lda];
          float a2 = a[offseta + i + (row + 2) * lda];
          float a3 = a[offseta + i + (row + 3) * lda];
          sum00 += a0 * b0;
          sum01 += a1 * b0;
          sum02 += a2 * b0;
          sum03 += a3 * b0;
          sum10 += a0 * b1;
          sum11 += a1 * b1;
          sum12 += a2 * b1;
          sum13 += a3 * b1;
          sum20 += a0 * b2;
          sum21 += a1 * b2;
          sum22 += a2 * b2;
          sum23 += a3 * b2;
          sum30 += a0 * b3;
          sum31 += a1 * b3;
          sum32 += a2 * b3;
          sum33 += a3 * b3;
        }
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (row + 0) + (col + 0) * ldc];
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01 + beta * c[offsetc + (row + 1) + (col + 0) * ldc];
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02 + beta * c[offsetc + (row + 2) + (col + 0) * ldc];
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03 + beta * c[offsetc + (row + 3) + (col + 0) * ldc];
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10 + beta * c[offsetc + (row + 0) + (col + 1) * ldc];
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (row + 1) + (col + 1) * ldc];
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12 + beta * c[offsetc + (row + 2) + (col + 1) * ldc];
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13 + beta * c[offsetc + (row + 3) + (col + 1) * ldc];
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20 + beta * c[offsetc + (row + 0) + (col + 2) * ldc];
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21 + beta * c[offsetc + (row + 1) + (col + 2) * ldc];
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (row + 2) + (col + 2) * ldc];
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23 + beta * c[offsetc + (row + 3) + (col + 2) * ldc];
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30 + beta * c[offsetc + (row + 0) + (col + 3) * ldc];
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31 + beta * c[offsetc + (row + 1) + (col + 3) * ldc];
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32 + beta * c[offsetc + (row + 2) + (col + 3) * ldc];
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (row + 3) + (col + 3) * ldc];
        } else {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00;
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum01;
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum02;
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum03;
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum10;
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11;
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum12;
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum13;
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum20;
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum21;
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22;
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum23;
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum30;
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum31;
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum32;
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33;
        }
      }
      for (; row < rowe; row += 1) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int i = is; i < ie; i += 1) {
          float aval = a[offseta + i + row * lda];
          sum0 += aval * b[offsetb + (col + 0) + i * ldb];
          sum1 += aval * b[offsetb + (col + 1) + i * ldb];
          sum2 += aval * b[offsetb + (col + 2) + i * ldb];
          sum3 += aval * b[offsetb + (col + 3) + i * ldb];
        }
        if (beta != 0.0f) {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0 + beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1 + beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2 + beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3 + beta * c[offsetc + row + (col + 3) * ldc];
        } else {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0;
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1;
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2;
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; col < cole; col += 1) {
      int row = rows;
      for (; row < loopBound(rowe, 4); row += 4) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (int i = is; i < ie; i += 1) {
          float bval = b[offsetb + col + i * ldb];
          sum0 += a[offseta + i + (row + 0) * lda] * bval;
          sum1 += a[offseta + i + (row + 1) * lda] * bval;
          sum2 += a[offseta + i + (row + 2) * lda] * bval;
          sum3 += a[offseta + i + (row + 3) * lda] * bval;
        }
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0 + beta * c[offsetc + (row + 0) + col * ldc];
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1 + beta * c[offsetc + (row + 1) + col * ldc];
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2 + beta * c[offsetc + (row + 2) + col * ldc];
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3 + beta * c[offsetc + (row + 3) + col * ldc];
        } else {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0;
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1;
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2;
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3;
        }
      }
      for (; row < rowe; row += 1) {
        float sum = 0.0f;
        for (int i = is; i < ie; i += 1) {
          sum += a[offseta + i + row * lda] * b[offsetb + col + i * ldb];
        }
        if (beta != 0.0f) {
          c[offsetc + row + col * ldc] = alpha * sum + beta * c[offsetc + row + col * ldc];
        } else {
          c[offsetc + row + col * ldc] = alpha * sum;
        }
      }
    }
  }

  public void dgemv(String trans, int m, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta, double[] y, int incy) {
    dgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  // y = alpha * A * x + beta * y
  public void dgemv(String trans, int m, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    if (!lsame("N", trans) && !lsame("T", trans) && !lsame("C", trans)) {
      throw illegalArgument("DGEMV", 1);
    }
    if (m < 0) {
      throw illegalArgument("DGEMV", 2);
    }
    if (n < 0) {
      throw illegalArgument("DGEMV", 3);
    }
    if (lda < Math.max(1, m)) {
      throw illegalArgument("DGEMV", 6);
    }
    if (incx == 0) {
      throw illegalArgument("DGEMV", 8);
    }
    if (incy == 0) {
      throw illegalArgument("DGEMV", 11);
    }
    if (m == 0 || n == 0) {
      return;
    }
    if (alpha == 0.0) {
      int len = lsame("N", trans) ? m : n;
      for (int i = 0, iy = incy < 0 ? (len - 1) * -incy : 0; i < len; i += 1, iy += incy) {
        if (beta != 0.0) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0;
        }
      }
    } else if (lsame("N", trans)) {
      dgemvN(m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("T", trans) || lsame("C", trans)) {
      dgemvT(m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void dgemvN(int m, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    // y = beta * y
    for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
      if (beta != 0.0) {
        y[offsety + iy] = beta * y[offsety + iy];
      } else {
        y[offsety + iy] = 0.0;
      }
    }
    // y += alpha * A * x
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      double alphax0 = alpha * x[offsetx + ix + incx * 0];
      double alphax1 = alpha * x[offsetx + ix + incx * 1];
      double alphax2 = alpha * x[offsetx + ix + incx * 2];
      double alphax3 = alpha * x[offsetx + ix + incx * 3];
      for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
        y[offsety + iy] += alphax0 * a[offseta + row + (col + 0) * lda]
                        +  alphax1 * a[offseta + row + (col + 1) * lda]
                        +  alphax2 * a[offseta + row + (col + 2) * lda]
                        +  alphax3 * a[offseta + row + (col + 3) * lda];
      }
    }
    for (; col < n; col += 1, ix += incx) {
      double alphax = alpha * x[offsetx + ix];
      for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
        y[offsety + iy] += alphax * a[offseta + row + col * lda];
      }
    }
  }

  protected void dgemvT(int m, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    int col = 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, iy += incy * 4) {
      double sum0 = 0.0;
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      for (int row = 0, ix = incx < 0 ? (m - 1) * -incx : 0; row < m; row += 1, ix += incx) {
        double xix = x[offsetx + ix];
        sum0 += xix * a[offseta + row + (col + 0) * lda];
        sum1 += xix * a[offseta + row + (col + 1) * lda];
        sum2 += xix * a[offseta + row + (col + 2) * lda];
        sum3 += xix * a[offseta + row + (col + 3) * lda];
      }
      if (beta != 0.0) {
        y[offsety + iy + incy * 0] = alpha * sum0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sum1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sum2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sum3 + beta * y[offsety + iy + incy * 3];
      } else {
        y[offsety + iy + incy * 0] = alpha * sum0;
        y[offsety + iy + incy * 1] = alpha * sum1;
        y[offsety + iy + incy * 2] = alpha * sum2;
        y[offsety + iy + incy * 3] = alpha * sum3;
      }
    }
    for (; col < n; col += 1, iy += incy) {
      double sum = 0.0;
      for (int row = 0, ix = incx < 0 ? (m - 1) * -incx : 0; row < m; row += 1, ix += incx) {
        sum += x[offsetx + ix] * a[offseta + row + col * lda];
      }
      if (beta != 0.0) {
        y[offsety + iy] = alpha * sum + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sum;
      }
    }
  }

  public void sgemv(String trans, int m, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta, float[] y, int incy) {
    sgemv(trans, m, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  // y = alpha * A * x + beta * y
  public void sgemv(String trans, int m, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    if (!lsame("N", trans) && !lsame("T", trans) && !lsame("C", trans)) {
      throw illegalArgument("SGEMV", 1);
    }
    if (m < 0) {
      throw illegalArgument("SGEMV", 2);
    }
    if (n < 0) {
      throw illegalArgument("SGEMV", 3);
    }
    if (lda < Math.max(1, m)) {
      throw illegalArgument("SGEMV", 6);
    }
    if (incx == 0) {
      throw illegalArgument("SGEMV", 8);
    }
    if (incy == 0) {
      throw illegalArgument("SGEMV", 11);
    }
    if (m == 0 || n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      int len = lsame("N", trans) ? m : n;
      for (int i = 0, iy = incy < 0 ? (len - 1) * -incy : 0; i < len; i += 1, iy += incy) {
        if (beta != 0.0f) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0f;
        }
      }
    } else if (lsame("N", trans)) {
      sgemvN(m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("T", trans) || lsame("C", trans)) {
      sgemvT(m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void sgemvN(int m, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    // y = beta * y
    for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
      if (beta != 0.0f) {
        y[offsety + iy] = beta * y[offsety + iy];
      } else {
        y[offsety + iy] = 0.0f;
      }
    }
    // y += alpha * A * x
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 8); col += 8, ix += incx * 8) {
      float alphax0 = alpha * x[offsetx + ix + incx * 0];
      float alphax1 = alpha * x[offsetx + ix + incx * 1];
      float alphax2 = alpha * x[offsetx + ix + incx * 2];
      float alphax3 = alpha * x[offsetx + ix + incx * 3];
      float alphax4 = alpha * x[offsetx + ix + incx * 4];
      float alphax5 = alpha * x[offsetx + ix + incx * 5];
      float alphax6 = alpha * x[offsetx + ix + incx * 6];
      float alphax7 = alpha * x[offsetx + ix + incx * 7];
      for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
        y[offsety + iy] += alphax0 * a[offseta + row + (col + 0) * lda]
                        +  alphax1 * a[offseta + row + (col + 1) * lda]
                        +  alphax2 * a[offseta + row + (col + 2) * lda]
                        +  alphax3 * a[offseta + row + (col + 3) * lda]
                        +  alphax4 * a[offseta + row + (col + 4) * lda]
                        +  alphax5 * a[offseta + row + (col + 5) * lda]
                        +  alphax6 * a[offseta + row + (col + 6) * lda]
                        +  alphax7 * a[offseta + row + (col + 7) * lda];
      }
    }
    for (; col < n; col += 1, ix += incx) {
      float alphax = alpha * x[offsetx + ix];
      for (int row = 0, iy = incy < 0 ? (m - 1) * -incy : 0; row < m; row += 1, iy += incy) {
        y[offsety + iy] += alphax * a[offseta + row + col * lda];
      }
    }
  }

  protected void sgemvT(int m, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    int col = 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 8); col += 8, iy += incy * 8) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      float sum3 = 0.0f;
      float sum4 = 0.0f;
      float sum5 = 0.0f;
      float sum6 = 0.0f;
      float sum7 = 0.0f;
      for (int row = 0, ix = incx < 0 ? (m - 1) * -incx : 0; row < m; row += 1, ix += incx) {
        sum0 += x[offsetx + ix] * a[offseta + row + (col + 0) * lda];
        sum1 += x[offsetx + ix] * a[offseta + row + (col + 1) * lda];
        sum2 += x[offsetx + ix] * a[offseta + row + (col + 2) * lda];
        sum3 += x[offsetx + ix] * a[offseta + row + (col + 3) * lda];
        sum4 += x[offsetx + ix] * a[offseta + row + (col + 4) * lda];
        sum5 += x[offsetx + ix] * a[offseta + row + (col + 5) * lda];
        sum6 += x[offsetx + ix] * a[offseta + row + (col + 6) * lda];
        sum7 += x[offsetx + ix] * a[offseta + row + (col + 7) * lda];
      }
      if (beta != 0.0f) {
        y[offsety + iy + incy * 0] = alpha * sum0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sum1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sum2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sum3 + beta * y[offsety + iy + incy * 3];
        y[offsety + iy + incy * 4] = alpha * sum4 + beta * y[offsety + iy + incy * 4];
        y[offsety + iy + incy * 5] = alpha * sum5 + beta * y[offsety + iy + incy * 5];
        y[offsety + iy + incy * 6] = alpha * sum6 + beta * y[offsety + iy + incy * 6];
        y[offsety + iy + incy * 7] = alpha * sum7 + beta * y[offsety + iy + incy * 7];
      } else {
        y[offsety + iy + incy * 0] = alpha * sum0;
        y[offsety + iy + incy * 1] = alpha * sum1;
        y[offsety + iy + incy * 2] = alpha * sum2;
        y[offsety + iy + incy * 3] = alpha * sum3;
        y[offsety + iy + incy * 4] = alpha * sum4;
        y[offsety + iy + incy * 5] = alpha * sum5;
        y[offsety + iy + incy * 6] = alpha * sum6;
        y[offsety + iy + incy * 7] = alpha * sum7;
      }
    }
    for (; col < n; col += 1, iy += incy) {
      float sum = 0.0f;
      for (int row = 0, ix = incx < 0 ? (m - 1) * -incx : 0; row < m; row += 1, ix += incx) {
        sum += x[offsetx + ix] * a[offseta + row + col * lda];
      }
      if (beta != 0.0f) {
        y[offsety + iy] = alpha * sum + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sum;
      }
    }
  }

  // A += alpha * x * y.t
  public void dger(int m, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda) {
    dger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
  }

  public void dger(int m, int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta, int lda) {
    if (m < 0) {
      throw illegalArgument("DGER", 1);
    }
    if (n < 0) {
      throw illegalArgument("DGER", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DGER", 5);
    }
    if (incy == 0) {
      throw illegalArgument("DGER", 7);
    }
    if (lda < Math.max(1, m)) {
      throw illegalArgument("DGER", 9);
    }
    if (m == 0 || n == 0) {
      return;
    }
    if (alpha != 0.0) {
      int col = 0, iy = incy < 0 ? (n - 1) * -incy : 0;
      for (; col < loopBound(n, 4); col += 4, iy += incy * 4) {
        double alphayiy0 = alpha * y[offsety + iy + incy * 0];
        double alphayiy1 = alpha * y[offsety + iy + incy * 1];
        double alphayiy2 = alpha * y[offsety + iy + incy * 2];
        double alphayiy3 = alpha * y[offsety + iy + incy * 3];
        int row = 0, jx = incx < 0 ? (n - 1) * -incx : 0;
        for (; row < m; row += 1, jx += incx) {
          double xjx = x[offsetx + jx];
          a[offseta + row + (col + 0) * lda] += alphayiy0 * xjx;
          a[offseta + row + (col + 1) * lda] += alphayiy1 * xjx;
          a[offseta + row + (col + 2) * lda] += alphayiy2 * xjx;
          a[offseta + row + (col + 3) * lda] += alphayiy3 * xjx;
        }
      }
      for (; col < n; col += 1, iy += incy) {
        double alphayiy = alpha * y[offsety + iy];
        int row = 0, jx = incx < 0 ? (n - 1) * -incx : 0;
        for (; row < m; row += 1, jx += incx) {
          a[offseta + row + col * lda] += alphayiy * x[offsetx + jx];
        }
      }
    }
  }

  public void sger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda) {
    sger(m, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
  }

  public void sger(int m, int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta, int lda) {
    if (m < 0) {
      throw illegalArgument("SGER", 1);
    }
    if (n < 0) {
      throw illegalArgument("SGER", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SGER", 5);
    }
    if (incy == 0) {
      throw illegalArgument("SGER", 7);
    }
    if (lda < Math.max(1, m)) {
      throw illegalArgument("SGER", 9);
    }
    if (m == 0 || n == 0) {
      return;
    }
    if (alpha != 0.0f) {
      int col = 0, iy = incy < 0 ? (n - 1) * -incy : 0;
      for (; col < loopBound(n, 4); col += 4, iy += incy * 4) {
        float alphayiy0 = alpha * y[offsety + iy + incy * 0];
        float alphayiy1 = alpha * y[offsety + iy + incy * 1];
        float alphayiy2 = alpha * y[offsety + iy + incy * 2];
        float alphayiy3 = alpha * y[offsety + iy + incy * 3];
        int row = 0, jx = incx < 0 ? (n - 1) * -incx : 0;
        for (; row < m; row += 1, jx += incx) {
          float xjx = x[offsetx + jx];
          a[offseta + row + (col + 0) * lda] += alphayiy0 * xjx;
          a[offseta + row + (col + 1) * lda] += alphayiy1 * xjx;
          a[offseta + row + (col + 2) * lda] += alphayiy2 * xjx;
          a[offseta + row + (col + 3) * lda] += alphayiy3 * xjx;
        }
      }
      for (; col < n; col += 1, iy += incy) {
        float alphayiy = alpha * y[offsety + iy];
        int row = 0, jx = incx < 0 ? (n - 1) * -incx : 0;
        for (; row < m; row += 1, jx += incx) {
          a[offseta + row + col * lda] += alphayiy * x[offsetx + jx];
        }
      }
    }
  }

  public double dnrm2(int n, double[] x, int incx) {
    return dnrm2(n, x, 0, incx);
  }

  public double dnrm2(int n, double[] x, int offsetx, int incx) {
    if (n <= 0) {
      return 0.0;
    }
    if (incx <= 0) {
      return 0.0;
    }
    if (n == 1) {
      return Math.abs(x[offsetx + 0]);
    }

    double sum = 0.0;
    for (int ix = 0; ix < n * incx; ix += incx) {
      sum += Math.pow(x[offsetx + ix], 2);
    }
    return Math.sqrt(sum);
  }

  public float snrm2(int n, float[] x, int incx) {
    return snrm2(n, x, 0, incx);
  }

  public float snrm2(int n, float[] x, int offsetx, int incx) {
    if (n <= 0) {
      return 0.0f;
    }
    if (incx <= 0) {
      return 0.0f;
    }
    if (n == 1) {
      return Math.abs(x[offsetx + 0]);
    }

    float sum = 0.0f;
    for (int ix = 0; ix < n * incx; ix += incx) {
      sum += (float)Math.pow(x[offsetx + ix], 2);
    }
    return (float)Math.sqrt(sum);
  }

  public void drot(int n, double[] x, int incx, double[] y, int incy, double c, double s) {
    drot(n, x, 0, incx, y, 0, incy, c, s);
  }

  public void drot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double c, double s) {
    if (n <= 0) {
      return;
    }

    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      double xold = x[offsetx + ix];
      double yold = y[offsety + iy];
      x[offsetx + ix] = c * xold + s * yold;
      y[offsety + iy] = c * yold - s * xold;
    }
  }

  public void srot(int n, float[] x, int incx, float[] y, int incy, float c, float s) {
    srot(n, x, 0, incx, y, 0, incy, c, s);
  }

  public void srot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float c, float s) {
    if (n <= 0) {
      return;
    }

    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      float xnew = c * x[offsetx + ix] + s * y[offsety + iy];
      float ynew = c * y[offsety + iy] - s * x[offsetx + ix];
      x[offsetx + ix] = xnew;
      y[offsety + iy] = ynew;
    }
  }

  public void drotg(org.netlib.util.doubleW da, org.netlib.util.doubleW db, org.netlib.util.doubleW c, org.netlib.util.doubleW s) {
    double scale = Math.abs(da.val) + Math.abs(db.val);
    if (scale == 0.0) {
      c.val = 1.0;
      s.val = 0.0;
      da.val = 0.0;
      db.val = 0.0;
    } else {
      double r = scale * Math.sqrt(Math.pow(da.val / scale, 2) + Math.pow(db.val / scale, 2))
                      * ((Math.abs(da.val) > Math.abs(db.val) ? da.val : db.val) >= 0.0 ? 1.0 : -1.0);
      c.val = da.val / r;
      s.val = db.val / r;
      double z = 1.0;
      if (Math.abs(da.val) > Math.abs(db.val)) {
        z = s.val;
      } else if (c.val != 0.0) {
        z = 1.0 / c.val;
      }
      da.val = r;
      db.val = z;
    }
  }

  public void srotg(org.netlib.util.floatW sa, org.netlib.util.floatW sb, org.netlib.util.floatW c, org.netlib.util.floatW s) {
    float scale = Math.abs(sa.val) + Math.abs(sb.val);
    if (scale == 0.0f) {
      c.val = 1.0f;
      s.val = 0.0f;
      sa.val = 0.0f;
      sb.val = 0.0f;
    } else {
      float r = (float)(scale * Math.sqrt(Math.pow(sa.val / scale, 2) + Math.pow(sb.val / scale, 2))
                              * ((Math.abs(sa.val) > Math.abs(sb.val) ? sa.val : sb.val) >= 0.0f ? 1.0 : -1.0));
      c.val = sa.val / r;
      s.val = sb.val / r;
      float z = 1.0f;
      if (Math.abs(sa.val) > Math.abs(sb.val)) {
        z = s.val;
      } else if (c.val != 0.0f) {
        z = 1.0f / c.val;
      }
      sa.val = r;
      sb.val = z;
    }
  }

  public void drotm(int n, double[] x, int incx, double[] y, int incy, double[] param) {
    drotm(n, x, 0, incx, y, 0, incy, param, 0);
  }

  public void drotm(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] param, int offsetparam) {
    f2j.drotm(n, x, offsetx, incx, y, offsety, incy, param, offsetparam);
  }

  public void srotm(int n, float[] x, int incx, float[] y, int incy, float[] param) {
    srotm(n, x, 0, incx, y, 0, incy, param, 0);
  }

  public void srotm(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] param, int offsetparam) {
    f2j.srotm(n, x, offsetx, incx, y, offsety, incy, param, offsetparam);
  }

  public void drotmg(org.netlib.util.doubleW dd1, org.netlib.util.doubleW dd2, org.netlib.util.doubleW dx1, double dy1, double[] param) {
    drotmg(dd1, dd2, dx1, dy1, param, 0);
  }

  public void drotmg(org.netlib.util.doubleW dd1, org.netlib.util.doubleW dd2, org.netlib.util.doubleW dx1, double dy1, double[] param, int offsetparam) {
    f2j.drotmg(dd1, dd2, dx1, dy1, param, offsetparam);
  }

  public void srotmg(org.netlib.util.floatW sd1, org.netlib.util.floatW sd2, org.netlib.util.floatW sx1, float sy1, float[] param) {
    srotmg(sd1, sd2, sx1, sy1, param, 0);
  }

  public void srotmg(org.netlib.util.floatW sd1, org.netlib.util.floatW sd2, org.netlib.util.floatW sx1, float sy1, float[] param, int offsetparam) {
    f2j.srotmg(sd1, sd2, sx1, sy1, param, offsetparam);
  }

  public void dsbmv(String uplo, int n, int k, double alpha, double[] a, int lda, double[] x, int incx, double beta, double[] y, int incy) {
    dsbmv(uplo, n, k, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void dsbmv(String uplo, int n, int k, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    f2j.dsbmv(uplo, n, k, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
  }

  public void ssbmv(String uplo, int n, int k, float alpha, float[] a, int lda, float[] x, int incx, float beta, float[] y, int incy) {
    ssbmv(uplo, n, k, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void ssbmv(String uplo, int n, int k, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    f2j.ssbmv(uplo, n, k, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
  }

  public void dscal(int n, double alpha, double[] x, int incx) {
    dscal(n, alpha, x, 0, incx);
  }

  // x = alpha * x
  public void dscal(int n, double alpha, double[] x, int offsetx, int incx) {
    if (n <= 0) {
      return;
    }
    if (incx <= 0) {
      return;
    }
    if (alpha == 1.0) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0; incx < 0 ? ix >= 0 : ix < n * incx; ix += incx) {
      x[offsetx + ix] *= alpha;
    }
  }

  public void sscal(int n, float alpha, float[] x, int incx) {
    sscal(n, alpha, x, 0, incx);
  }

  // x = alpha * x
  public void sscal(int n, float alpha, float[] x, int offsetx, int incx) {
    if (n <= 0) {
      return;
    }
    if (incx <= 0) {
      return;
    }
    if (alpha == 1.0f) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0; incx < 0 ? ix >= 0 : ix < n * incx; ix += incx) {
      x[offsetx + ix] *= alpha;
    }
  }

  public void dspmv(String uplo, int n, double alpha, double[] a, double[] x, int incx, double beta, double[] y, int incy) {
    dspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
  }

  // y = alpha * a * x + beta * y
  public void dspmv(String uplo, int n, double alpha, double[] a, int offseta, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSPMV", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSPMV", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DSPMV", 6);
    }
    if (incy == 0) {
      throw illegalArgument("DSPMV", 9);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0;
        }
      }
    } else if (lsame("U", uplo)) {
      dspmvU(n, alpha, a, offseta, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("L", uplo)) {
      dspmvL(n, alpha, a, offseta, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void dspmvU(int n, double alpha, double[] a, int offseta, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double sumiy0 = 0.0;
      double sumiy1 = 0.0;
      double sumiy2 = 0.0;
      double sumiy3 = 0.0;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        double a0 = a[offseta + row + (col + 0) * ((col + 0) + 1) / 2];
        double a1 = a[offseta + row + (col + 1) * ((col + 1) + 1) / 2];
        double a2 = a[offseta + row + (col + 2) * ((col + 2) + 1) / 2];
        double a3 = a[offseta + row + (col + 3) * ((col + 3) + 1) / 2];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        double xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      double a00 = a[offseta + (row + 0) + (col + 0) * ((col + 0) + 1) / 2];
      double a01 = a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2];
      double a02 = a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2];
      double a03 = a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2];
      double a11 = a[offseta + (row + 1) + (col + 1) * ((col + 1) + 1) / 2];
      double a12 = a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2];
      double a13 = a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2];
      double a22 = a[offseta + (row + 2) + (col + 2) * ((col + 2) + 1) / 2];
      double a23 = a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2];
      double a33 = a[offseta + (row + 3) + (col + 3) * ((col + 3) + 1) / 2];
      double xjx0 = x[offsetx + jx + incx * 0];
      double xjx1 = x[offsetx + jx + incx * 1];
      double xjx2 = x[offsetx + jx + incx * 2];
      double xjx3 = x[offsetx + jx + incx * 3];
      sumiy0 += xjx0 * a00
             +  xjx1 * a01
             +  xjx2 * a02
             +  xjx3 * a03;
      sumiy1 += xjx0 * a01
             +  xjx1 * a11
             +  xjx2 * a12
             +  xjx3 * a13;
      sumiy2 += xjx0 * a02
             +  xjx1 * a12
             +  xjx2 * a22
             +  xjx3 * a23;
      sumiy3 += xjx0 * a03
             +  xjx1 * a13
             +  xjx2 * a23
             +  xjx3 * a33;
      if (beta != 0.0) {
        y[offsety + iy + incy * 0] = alpha * sumiy0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sumiy1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sumiy2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sumiy3 + beta * y[offsety + iy + incy * 3];
      } else {
        y[offsety + iy + incy * 0] = alpha * sumiy0;
        y[offsety + iy + incy * 1] = alpha * sumiy1;
        y[offsety + iy + incy * 2] = alpha * sumiy2;
        y[offsety + iy + incy * 3] = alpha * sumiy3;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double sumiy = 0.0;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * (col + 1) / 2];
        sumiy += x[offsetx + jx] * a[offseta + row + col * (col + 1) / 2];
      }
      sumiy += x[offsetx + jx] * a[offseta + row + col * (col + 1) / 2];
      if (beta != 0.0) {
        y[offsety + iy] = alpha * sumiy + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sumiy;
      }
    }
  }

  protected void dspmvL(int n, double alpha, double[] a, int offseta, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    // y = beta * y
    if (beta != 1.0) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0;
        }
      }
    }
    // y += alpha * A * x
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double sumiy0 = 0.0;
      double sumiy1 = 0.0;
      double sumiy2 = 0.0;
      double sumiy3 = 0.0;
      double a00 = a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      double a10 = a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      double a11 = a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      double a20 = a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      double a21 = a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      double a22 = a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2];
      double a30 = a[offseta + /*row=*/(col + 3) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      double a31 = a[offseta + /*row=*/(col + 3) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      double a32 = a[offseta + /*row=*/(col + 3) + (col + 2) * (2 * n - (col + 2) - 1) / 2];
      double a33 = a[offseta + /*row=*/(col + 3) + (col + 3) * (2 * n - (col + 3) - 1) / 2];
      double x0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      double x1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      double x2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      double x3 = x[offsetx + (incx < 0 ? (n - (col + 3) - 1) * -incx : (col + 3) * incx)];
      sumiy0 += x0 * a00
             +  x1 * a10
             +  x2 * a20
             +  x3 * a30;
      sumiy1 += x0 * a10
             +  x1 * a11
             +  x2 * a21
             +  x3 * a31;
      sumiy2 += x0 * a20
             +  x1 * a21
             +  x2 * a22
             +  x3 * a32;
      sumiy3 += x0 * a30
             +  x1 * a31
             +  x2 * a32
             +  x3 * a33;
      int row = col + 4, jx = incx < 0 ? (n - (col + 4) - 1) * -incx : (col + 4) * incx, jy = incy < 0 ? (n - (col + 4) - 1) * -incy : (col + 4) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        double a0 = a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2];
        double a1 = a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2];
        double a2 = a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2];
        double a3 = a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        double xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      y[offsety + iy + incy * 0] += alpha * sumiy0;
      y[offsety + iy + incy * 1] += alpha * sumiy1;
      y[offsety + iy + incy * 2] += alpha * sumiy2;
      y[offsety + iy + incy * 3] += alpha * sumiy3;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double sumiy = 0.0;
      sumiy += x[offsetx + (incx < 0 ? (n - col - 1) * -incx : col * incx)] * a[offseta + /*row=*/col + col * (2 * n - col - 1) / 2];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * (2 * n - col - 1) / 2];
        sumiy += x[offsetx + jx] * a[offseta + row + col * (2 * n - col - 1) / 2];
      }
      y[offsety + iy] += alpha * sumiy;
    }
  }

  public void sspmv(String uplo, int n, float alpha, float[] a, float[] x, int incx, float beta, float[] y, int incy) {
    sspmv(uplo, n, alpha, a, 0, x, 0, incx, beta, y, 0, incy);
  }

  public void sspmv(String uplo, int n, float alpha, float[] a, int offseta, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSPMV", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSPMV", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SSPMV", 6);
    }
    if (incy == 0) {
      throw illegalArgument("SSPMV", 9);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0f) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0f;
        }
      }
    } else if (lsame("U", uplo)) {
      sspmvU(n, alpha, a, offseta, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("L", uplo)) {
      sspmvL(n, alpha, a, offseta, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void sspmvU(int n, float alpha, float[] a, int offseta, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float sumiy0 = 0.0f;
      float sumiy1 = 0.0f;
      float sumiy2 = 0.0f;
      float sumiy3 = 0.0f;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        float a0 = a[offseta + row + (col + 0) * ((col + 0) + 1) / 2];
        float a1 = a[offseta + row + (col + 1) * ((col + 1) + 1) / 2];
        float a2 = a[offseta + row + (col + 2) * ((col + 2) + 1) / 2];
        float a3 = a[offseta + row + (col + 3) * ((col + 3) + 1) / 2];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        float xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      float a00 = a[offseta + (row + 0) + (col + 0) * ((col + 0) + 1) / 2];
      float a01 = a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2];
      float a02 = a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2];
      float a03 = a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2];
      float a11 = a[offseta + (row + 1) + (col + 1) * ((col + 1) + 1) / 2];
      float a12 = a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2];
      float a13 = a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2];
      float a22 = a[offseta + (row + 2) + (col + 2) * ((col + 2) + 1) / 2];
      float a23 = a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2];
      float a33 = a[offseta + (row + 3) + (col + 3) * ((col + 3) + 1) / 2];
      float xjx0 = x[offsetx + jx + incx * 0];
      float xjx1 = x[offsetx + jx + incx * 1];
      float xjx2 = x[offsetx + jx + incx * 2];
      float xjx3 = x[offsetx + jx + incx * 3];
      sumiy0 += xjx0 * a00
             +  xjx1 * a01
             +  xjx2 * a02
             +  xjx3 * a03;
      sumiy1 += xjx0 * a01
             +  xjx1 * a11
             +  xjx2 * a12
             +  xjx3 * a13;
      sumiy2 += xjx0 * a02
             +  xjx1 * a12
             +  xjx2 * a22
             +  xjx3 * a23;
      sumiy3 += xjx0 * a03
             +  xjx1 * a13
             +  xjx2 * a23
             +  xjx3 * a33;
      if (beta != 0.0f) {
        y[offsety + iy + incy * 0] = alpha * sumiy0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sumiy1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sumiy2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sumiy3 + beta * y[offsety + iy + incy * 3];
      } else {
        y[offsety + iy + incy * 0] = alpha * sumiy0;
        y[offsety + iy + incy * 1] = alpha * sumiy1;
        y[offsety + iy + incy * 2] = alpha * sumiy2;
        y[offsety + iy + incy * 3] = alpha * sumiy3;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float sumiy = 0.0f;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * (col + 1) / 2];
        sumiy += x[offsetx + jx] * a[offseta + row + col * (col + 1) / 2];
      }
      sumiy += x[offsetx + jx] * a[offseta + row + col * (col + 1) / 2];
      if (beta != 0.0f) {
        y[offsety + iy] = alpha * sumiy + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sumiy;
      }
    }
  }

  protected void sspmvL(int n, float alpha, float[] a, int offseta, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    // y = beta * y
    if (beta != 1.0f) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0f) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0f;
        }
      }
    }
    // y += alpha * A * x
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float sumiy0 = 0.0f;
      float sumiy1 = 0.0f;
      float sumiy2 = 0.0f;
      float sumiy3 = 0.0f;
      float a00 = a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      float a10 = a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      float a11 = a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      float a20 = a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      float a21 = a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      float a22 = a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2];
      float a30 = a[offseta + /*row=*/(col + 3) + (col + 0) * (2 * n - (col + 0) - 1) / 2];
      float a31 = a[offseta + /*row=*/(col + 3) + (col + 1) * (2 * n - (col + 1) - 1) / 2];
      float a32 = a[offseta + /*row=*/(col + 3) + (col + 2) * (2 * n - (col + 2) - 1) / 2];
      float a33 = a[offseta + /*row=*/(col + 3) + (col + 3) * (2 * n - (col + 3) - 1) / 2];
      float x0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      float x1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      float x2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      float x3 = x[offsetx + (incx < 0 ? (n - (col + 3) - 1) * -incx : (col + 3) * incx)];
      sumiy0 += x0 * a00
             +  x1 * a10
             +  x2 * a20
             +  x3 * a30;
      sumiy1 += x0 * a10
             +  x1 * a11
             +  x2 * a21
             +  x3 * a31;
      sumiy2 += x0 * a20
             +  x1 * a21
             +  x2 * a22
             +  x3 * a32;
      sumiy3 += x0 * a30
             +  x1 * a31
             +  x2 * a32
             +  x3 * a33;
      int row = col + 4, jx = incx < 0 ? (n - (col + 4) - 1) * -incx : (col + 4) * incx, jy = incy < 0 ? (n - (col + 4) - 1) * -incy : (col + 4) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        float a0 = a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2];
        float a1 = a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2];
        float a2 = a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2];
        float a3 = a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        float xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      y[offsety + iy + incy * 0] += alpha * sumiy0;
      y[offsety + iy + incy * 1] += alpha * sumiy1;
      y[offsety + iy + incy * 2] += alpha * sumiy2;
      y[offsety + iy + incy * 3] += alpha * sumiy3;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float sumiy = 0.0f;
      sumiy += x[offsetx + (incx < 0 ? (n - col - 1) * -incx : col * incx)] * a[offseta + /*row=*/col + col * (2 * n - col - 1) / 2];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * (2 * n - col - 1) / 2];
        sumiy += x[offsetx + jx] * a[offseta + row + col * (2 * n - col - 1) / 2];
      }
      y[offsety + iy] += alpha * sumiy;
    }
  }

  public void dspr(String uplo, int n, double alpha, double[] x, int incx, double[] a) {
    dspr(uplo, n, alpha, x, 0, incx, a, 0);
  }

  // a += alpha * x * x.t
  public void dspr(String uplo, int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSPR", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSPR", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DSPR", 5);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      // do nothing
    } else if (lsame("U", uplo)) {
      dsprU(n, alpha, x, offsetx, incx, a, offseta);
    } else if (lsame("L", uplo)) {
      dsprL(n, alpha, x, offsetx, incx, a, offseta);
    }
  }

  protected void dsprU(int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * ((col + 0) + 1) / 2] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx;
      }
      double xjx0 = x[offsetx + jx + incx * 0];
      a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx0;
      double xjx1 = x[offsetx + jx + incx * 1];
      a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx1;
      double xjx2 = x[offsetx + jx + incx * 2];
      a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx) {
      double alphaxix = alpha * x[offsetx + ix];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        a[offseta + row + col * (col + 1) / 2] += alphaxix * x[offsetx + jx];
      }
    }
  }

  protected void dsprL(int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx0;
      double xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx1;
      double xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx;
      for (; row < n; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2] += alphaxix3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx) {
      double alphaxix = alpha * x[offsetx + ix];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx;
      for (; row < n; row += 1, jx += incx) {
        a[offseta + row + col * (2 * n - col - 1) / 2] += alphaxix * x[offsetx + jx];
      }
    }
  }

  public void sspr(String uplo, int n, float alpha, float[] x, int incx, float[] a) {
    sspr(uplo, n, alpha, x, 0, incx, a, 0);
  }

  // a += alpha * x * x.t
  public void sspr(String uplo, int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSPR", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSPR", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SSPR", 5);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      // do nothing
    } else if (lsame("U", uplo)) {
      ssprU(n, alpha, x, offsetx, incx, a, offseta);
    } else if (lsame("L", uplo)) {
      ssprL(n, alpha, x, offsetx, incx, a, offseta);
    }
  }

  protected void ssprU(int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * ((col + 0) + 1) / 2] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx;
      }
      float xjx0 = x[offsetx + jx + incx * 0];
      a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx0;
      float xjx1 = x[offsetx + jx + incx * 1];
      a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx1;
      float xjx2 = x[offsetx + jx + incx * 2];
      a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx) {
      float alphaxix = alpha * x[offsetx + ix];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        a[offseta + row + col * (col + 1) / 2] += alphaxix * x[offsetx + jx];
      }
    }
  }

  protected void ssprL(int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx0;
      float xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx1;
      float xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx;
      for (; row < n; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2] += alphaxix3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx) {
      float alphaxix = alpha * x[offsetx + ix];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx;
      for (; row < n; row += 1, jx += incx) {
        a[offseta + row + col * (2 * n - col - 1) / 2] += alphaxix * x[offsetx + jx];
      }
    }
  }

  public void dspr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a) {
    dspr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0);
  }

  // a += alpha * x * y.t + alpha * y * x.t
  public void dspr2(String uplo, int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSPR2", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSPR2", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DSPR2", 5);
    }
    if (incy == 0) {
      throw illegalArgument("DSPR2", 7);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      // do nothing
    } else if (lsame("U", uplo)) {
      dspr2U(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta);
    } else if (lsame("L", uplo)) {
      dspr2L(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta);
    }
  }

  protected void dspr2U(int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double alphayiy0 = alpha * y[offsety + iy + incy * 0];
      double alphayiy1 = alpha * y[offsety + iy + incy * 1];
      double alphayiy2 = alpha * y[offsety + iy + incy * 2];
      double alphayiy3 = alpha * y[offsety + iy + incy * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        double xjx = x[offsetx + jx];
        double yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * ((col + 0) + 1) / 2] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
      double xjx0 = x[offsetx + jx + incx * 0];
      double yjy0 = y[offsety + jy + incy * 0];
      a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * yjy0 + alphayiy1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy0 + alphayiy2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy0 + alphayiy3 * xjx0;
      double xjx1 = x[offsetx + jx + incx * 1];
      double yjy1 = y[offsety + jy + incy * 1];
      a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy1 + alphayiy2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy1 + alphayiy3 * xjx1;
      double xjx2 = x[offsetx + jx + incx * 2];
      double yjy2 = y[offsety + jy + incy * 2];
      a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy2 + alphayiy3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double alphayiy = alpha * y[offsety + iy];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * (col + 1) / 2] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  protected void dspr2L(int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double alphayiy0 = alpha * y[offsety + iy + incy * 0];
      double alphayiy1 = alpha * y[offsety + iy + incy * 1];
      double alphayiy2 = alpha * y[offsety + iy + incy * 2];
      double alphayiy3 = alpha * y[offsety + iy + incy * 3];
      double xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      double yjy0 = y[offsety + (incy < 0 ? (n - (col + 0) - 1) * -incy : (col + 0) * incy)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy0 + alphayiy0 * xjx0;
      double xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      double yjy1 = y[offsety + (incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy1 + alphayiy0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy1 + alphayiy1 * xjx1;
      double xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      double yjy2 = y[offsety + (incy < 0 ? (n - (col + 2) - 1) * -incy : (col + 2) * incy)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy2 + alphayiy0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy2 + alphayiy1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * yjy2 + alphayiy2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < n; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        double yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double alphayiy = alpha * y[offsety + iy];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * (2 * n - col - 1) / 2] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  public void sspr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a) {
    sspr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0);
  }

  // a += alpha * x * y.t + alpha * y * x.t
  public void sspr2(String uplo, int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSPR2", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSPR2", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SSPR2", 5);
    }
    if (incy == 0) {
      throw illegalArgument("SSPR2", 7);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      // do nothing
    } else if (lsame("U", uplo)) {
      sspr2U(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta);
    } else if (lsame("L", uplo)) {
      sspr2L(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta);
    }
  }

  protected void sspr2U(int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float alphayiy0 = alpha * y[offsety + iy + incy * 0];
      float alphayiy1 = alpha * y[offsety + iy + incy * 1];
      float alphayiy2 = alpha * y[offsety + iy + incy * 2];
      float alphayiy3 = alpha * y[offsety + iy + incy * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        float xjx = x[offsetx + jx];
        float yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * ((col + 0) + 1) / 2] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
      float xjx0 = x[offsetx + jx + incx * 0];
      float yjy0 = y[offsety + jy + incy * 0];
      a[offseta + (row + 0) + (col + 1) * ((col + 1) + 1) / 2] += alphaxix1 * yjy0 + alphayiy1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy0 + alphayiy2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy0 + alphayiy3 * xjx0;
      float xjx1 = x[offsetx + jx + incx * 1];
      float yjy1 = y[offsety + jy + incy * 1];
      a[offseta + (row + 1) + (col + 2) * ((col + 2) + 1) / 2] += alphaxix2 * yjy1 + alphayiy2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy1 + alphayiy3 * xjx1;
      float xjx2 = x[offsetx + jx + incx * 2];
      float yjy2 = y[offsety + jy + incy * 2];
      a[offseta + (row + 2) + (col + 3) * ((col + 3) + 1) / 2] += alphaxix3 * yjy2 + alphayiy3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float alphayiy = alpha * y[offsety + iy];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * (col + 1) / 2] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  protected void sspr2L(int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float alphayiy0 = alpha * y[offsety + iy + incy * 0];
      float alphayiy1 = alpha * y[offsety + iy + incy * 1];
      float alphayiy2 = alpha * y[offsety + iy + incy * 2];
      float alphayiy3 = alpha * y[offsety + iy + incy * 3];
      float xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      float yjy0 = y[offsety + (incy < 0 ? (n - (col + 0) - 1) * -incy : (col + 0) * incy)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy0 + alphayiy0 * xjx0;
      float xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      float yjy1 = y[offsety + (incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy1 + alphayiy0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy1 + alphayiy1 * xjx1;
      float xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      float yjy2 = y[offsety + (incy < 0 ? (n - (col + 2) - 1) * -incy : (col + 2) * incy)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy2 + alphayiy0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy2 + alphayiy1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * yjy2 + alphayiy2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < n; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        float yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * (2 * n - (col + 0) - 1) / 2] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * (2 * n - (col + 1) - 1) / 2] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * (2 * n - (col + 2) - 1) / 2] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * (2 * n - (col + 3) - 1) / 2] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float alphayiy = alpha * y[offsety + iy];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * (2 * n - col - 1) / 2] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  public void dswap(int n, double[] x, int incx, double[] y, int incy) {
    dswap(n, x, 0, incx, y, 0, incy);
  }

  public void dswap(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      double tmp = y[offsety + iy];
      y[offsety + iy] = x[offsetx + ix];
      x[offsetx + ix] = tmp;
    }
  }

  public void sswap(int n, float[] x, int incx, float[] y, int incy) {
    sswap(n, x, 0, incx, y, 0, incy);
  }

  public void sswap(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy) {
    if (n <= 0) {
      return;
    }
    for (int ix = incx < 0 ? (n - 1) * -incx : 0,
             iy = incy < 0 ? (n - 1) * -incy : 0;
         (incx < 0 ? ix >= 0 : ix < n * incx)
          && (incy < 0 ? iy >= 0 : iy < n * incy);
         ix += incx, iy += incy) {
      float tmp = y[offsety + iy];
      y[offsety + iy] = x[offsetx + ix];
      x[offsetx + ix] = tmp;
    }
  }

  public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int lda, double[] b, int ldb, double beta, double[] c, int ldc) {
    dsymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  public void dsymm(String side, String uplo, int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    if (!lsame("L", side) && !lsame("R", side)) {
      throw illegalArgument("DSYMM", 1);
    }
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSYMM", 2);
    }
    if (m < 0) {
      throw illegalArgument("DSYMM", 3);
    }
    if (n < 0) {
      throw illegalArgument("DSYMM", 4);
    }
    if (lda < Math.max(1, lsame("L", side) ? m : n)) {
      throw illegalArgument("DSYMM", 7);
    }
    if (ldb < Math.max(1, m)) {
      throw illegalArgument("DSYMM", 9);
    }
    if (ldc < Math.max(1, m)) {
      throw illegalArgument("DSYMM", 12);
    }
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) {
      return;
    }
    if (alpha == 0.0) {
      // C := beta*C
      int col = 0;
      for (; col < loopBound(n, 4); col += 4) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
        }
      }
      for (; col < n; col += 1) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + col * ldc] = beta * c[offsetc + row + col * ldc];
        }
      }
    } else if (lsame("L", side) && lsame("U", uplo)) {
      dsymmLU(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("L", side) && lsame("L", uplo)) {
      dsymmLL(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("R", side) && lsame("U", uplo)) {
      dsymmRU(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("R", side) && lsame("L", uplo)) {
      dsymmRL(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    }
  }

  protected void dsymmLU(int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    // C := alpha*A*B + beta*C
    int col = 0;
    for (; col < loopBound(n, 4); col += 4) {
      int row = 0;
      for (; row < loopBound(m, 4); row += 4) {
        double sum00 = 0.0;
        double sum10 = 0.0;
        double sum20 = 0.0;
        double sum30 = 0.0;
        double sum01 = 0.0;
        double sum11 = 0.0;
        double sum21 = 0.0;
        double sum31 = 0.0;
        double sum02 = 0.0;
        double sum12 = 0.0;
        double sum22 = 0.0;
        double sum32 = 0.0;
        double sum03 = 0.0;
        double sum13 = 0.0;
        double sum23 = 0.0;
        double sum33 = 0.0;
        double alphab00 = alpha * b[offsetb + (row + 0) + (col + 0) * ldb];
        double alphab10 = alpha * b[offsetb + (row + 1) + (col + 0) * ldb];
        double alphab20 = alpha * b[offsetb + (row + 2) + (col + 0) * ldb];
        double alphab30 = alpha * b[offsetb + (row + 3) + (col + 0) * ldb];
        double alphab01 = alpha * b[offsetb + (row + 0) + (col + 1) * ldb];
        double alphab11 = alpha * b[offsetb + (row + 1) + (col + 1) * ldb];
        double alphab21 = alpha * b[offsetb + (row + 2) + (col + 1) * ldb];
        double alphab31 = alpha * b[offsetb + (row + 3) + (col + 1) * ldb];
        double alphab02 = alpha * b[offsetb + (row + 0) + (col + 2) * ldb];
        double alphab12 = alpha * b[offsetb + (row + 1) + (col + 2) * ldb];
        double alphab22 = alpha * b[offsetb + (row + 2) + (col + 2) * ldb];
        double alphab32 = alpha * b[offsetb + (row + 3) + (col + 2) * ldb];
        double alphab03 = alpha * b[offsetb + (row + 0) + (col + 3) * ldb];
        double alphab13 = alpha * b[offsetb + (row + 1) + (col + 3) * ldb];
        double alphab23 = alpha * b[offsetb + (row + 2) + (col + 3) * ldb];
        double alphab33 = alpha * b[offsetb + (row + 3) + (col + 3) * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          double a0 = a[offseta + i + (row + 0) * lda];
          double a1 = a[offseta + i + (row + 1) * lda];
          double a2 = a[offseta + i + (row + 2) * lda];
          double a3 = a[offseta + i + (row + 3) * lda];
          c[offsetc + i + (col + 0) * ldc] += alphab00 * a0
                                           +  alphab10 * a1
                                           +  alphab20 * a2
                                           +  alphab30 * a3;
          c[offsetc + i + (col + 1) * ldc] += alphab01 * a0
                                           +  alphab11 * a1
                                           +  alphab21 * a2
                                           +  alphab31 * a3;
          c[offsetc + i + (col + 2) * ldc] += alphab02 * a0
                                           +  alphab12 * a1
                                           +  alphab22 * a2
                                           +  alphab32 * a3;
          c[offsetc + i + (col + 3) * ldc] += alphab03 * a0
                                           +  alphab13 * a1
                                           +  alphab23 * a2
                                           +  alphab33 * a3;
          double b0 = b[offsetb + i + (col + 0) * ldb];
          double b1 = b[offsetb + i + (col + 1) * ldb];
          double b2 = b[offsetb + i + (col + 2) * ldb];
          double b3 = b[offsetb + i + (col + 3) * ldb];
          sum00 += a0 * b0;
          sum10 += a1 * b0;
          sum20 += a2 * b0;
          sum30 += a3 * b0;
          sum01 += a0 * b1;
          sum11 += a1 * b1;
          sum21 += a2 * b1;
          sum31 += a3 * b1;
          sum02 += a0 * b2;
          sum12 += a1 * b2;
          sum22 += a2 * b2;
          sum32 += a3 * b2;
          sum03 += a0 * b3;
          sum13 += a1 * b3;
          sum23 += a2 * b3;
          sum33 += a3 * b3;
        }
        double a00 = a[offseta + (i + 0) + (row + 0) * lda];
        double a01 = a[offseta + (i + 0) + (row + 1) * lda];
        double a02 = a[offseta + (i + 0) + (row + 2) * lda];
        double a03 = a[offseta + (i + 0) + (row + 3) * lda];
        double a11 = a[offseta + (i + 1) + (row + 1) * lda];
        double a12 = a[offseta + (i + 1) + (row + 2) * lda];
        double a13 = a[offseta + (i + 1) + (row + 3) * lda];
        double a22 = a[offseta + (i + 2) + (row + 2) * lda];
        double a23 = a[offseta + (i + 2) + (row + 3) * lda];
        double a33 = a[offseta + (i + 3) + (row + 3) * lda];
        double b00 = b[offsetb + (i + 0) + (col + 0) * ldb];
        double b10 = b[offsetb + (i + 1) + (col + 0) * ldb];
        double b20 = b[offsetb + (i + 2) + (col + 0) * ldb];
        double b30 = b[offsetb + (i + 3) + (col + 0) * ldb];
        double b01 = b[offsetb + (i + 0) + (col + 1) * ldb];
        double b11 = b[offsetb + (i + 1) + (col + 1) * ldb];
        double b21 = b[offsetb + (i + 2) + (col + 1) * ldb];
        double b31 = b[offsetb + (i + 3) + (col + 1) * ldb];
        double b02 = b[offsetb + (i + 0) + (col + 2) * ldb];
        double b12 = b[offsetb + (i + 1) + (col + 2) * ldb];
        double b22 = b[offsetb + (i + 2) + (col + 2) * ldb];
        double b32 = b[offsetb + (i + 3) + (col + 2) * ldb];
        double b03 = b[offsetb + (i + 0) + (col + 3) * ldb];
        double b13 = b[offsetb + (i + 1) + (col + 3) * ldb];
        double b23 = b[offsetb + (i + 2) + (col + 3) * ldb];
        double b33 = b[offsetb + (i + 3) + (col + 3) * ldb];
        sum00 += a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
        sum10 += a01 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
        sum20 += a02 * b00 + a12 * b10 + a22 * b20 + a23 * b30;
        sum30 += a03 * b00 + a13 * b10 + a23 * b20 + a33 * b30;
        sum01 += a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
        sum11 += a01 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
        sum21 += a02 * b01 + a12 * b11 + a22 * b21 + a23 * b31;
        sum31 += a03 * b01 + a13 * b11 + a23 * b21 + a33 * b31;
        sum02 += a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
        sum12 += a01 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
        sum22 += a02 * b02 + a12 * b12 + a22 * b22 + a23 * b32;
        sum32 += a03 * b02 + a13 * b12 + a23 * b22 + a33 * b32;
        sum03 += a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
        sum13 += a01 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
        sum23 += a02 * b03 + a12 * b13 + a22 * b23 + a23 * b33;
        sum33 += a03 * b03 + a13 * b13 + a23 * b23 + a33 * b33;
        if (beta != 0.0) {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (row + 0) + (col + 0) * ldc];
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum10 + beta * c[offsetc + (row + 1) + (col + 0) * ldc];
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum20 + beta * c[offsetc + (row + 2) + (col + 0) * ldc];
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum30 + beta * c[offsetc + (row + 3) + (col + 0) * ldc];
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum01 + beta * c[offsetc + (row + 0) + (col + 1) * ldc];
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (row + 1) + (col + 1) * ldc];
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum21 + beta * c[offsetc + (row + 2) + (col + 1) * ldc];
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum31 + beta * c[offsetc + (row + 3) + (col + 1) * ldc];
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum02 + beta * c[offsetc + (row + 0) + (col + 2) * ldc];
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum12 + beta * c[offsetc + (row + 1) + (col + 2) * ldc];
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (row + 2) + (col + 2) * ldc];
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum32 + beta * c[offsetc + (row + 3) + (col + 2) * ldc];
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum03 + beta * c[offsetc + (row + 0) + (col + 3) * ldc];
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum13 + beta * c[offsetc + (row + 1) + (col + 3) * ldc];
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum23 + beta * c[offsetc + (row + 2) + (col + 3) * ldc];
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (row + 3) + (col + 3) * ldc];
        } else {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00;
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum10;
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum20;
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum30;
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum01;
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11;
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum21;
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum31;
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum02;
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum12;
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22;
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum32;
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum03;
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum13;
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum23;
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33;
        }
      }
      for (; row < m; row += 1) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        double alphab0 = alpha * b[offsetb + row + (col + 0) * ldb];
        double alphab1 = alpha * b[offsetb + row + (col + 1) * ldb];
        double alphab2 = alpha * b[offsetb + row + (col + 2) * ldb];
        double alphab3 = alpha * b[offsetb + row + (col + 3) * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          double a0 = a[offseta + i + row * lda];
          c[offsetc + i + (col + 0) * ldc] += alphab0 * a0;
          c[offsetc + i + (col + 1) * ldc] += alphab1 * a0;
          c[offsetc + i + (col + 2) * ldc] += alphab2 * a0;
          c[offsetc + i + (col + 3) * ldc] += alphab3 * a0;
          sum0 += b[offsetb + i + (col + 0) * ldb] * a0;
          sum1 += b[offsetb + i + (col + 1) * ldb] * a0;
          sum2 += b[offsetb + i + (col + 2) * ldb] * a0;
          sum3 += b[offsetb + i + (col + 3) * ldb] * a0;
        }
        double a0 = a[offseta + i + row * lda];
        sum0 += b[offsetb + i + (col + 0) * ldb] * a0;
        sum1 += b[offsetb + i + (col + 1) * ldb] * a0;
        sum2 += b[offsetb + i + (col + 2) * ldb] * a0;
        sum3 += b[offsetb + i + (col + 3) * ldb] * a0;
        if (beta != 0.0) {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0 + beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1 + beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2 + beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3 + beta * c[offsetc + row + (col + 3) * ldc];
        } else {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0;
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1;
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2;
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; col < n; col += 1) {
      int row = 0;
      for (; row < loopBound(m, 4); row += 4) {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        double alphab0 = alpha * b[offsetb + (row + 0) + col * ldb];
        double alphab1 = alpha * b[offsetb + (row + 1) + col * ldb];
        double alphab2 = alpha * b[offsetb + (row + 2) + col * ldb];
        double alphab3 = alpha * b[offsetb + (row + 3) + col * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          double a0 = a[offseta + i + (row + 0) * lda];
          double a1 = a[offseta + i + (row + 1) * lda];
          double a2 = a[offseta + i + (row + 2) * lda];
          double a3 = a[offseta + i + (row + 3) * lda];
          c[offsetc + i + col * ldc] += alphab0 * a0
                                     +  alphab1 * a1
                                     +  alphab2 * a2
                                     +  alphab3 * a3;
          double b0 = b[offsetb + i + col * ldb];
          sum0 += b0 * a0;
          sum1 += b0 * a1;
          sum2 += b0 * a2;
          sum3 += b0 * a3;
        }
        double a00 = a[offseta + (i + 0) + (row + 0) * lda];
        double a01 = a[offseta + (i + 0) + (row + 1) * lda];
        double a02 = a[offseta + (i + 0) + (row + 2) * lda];
        double a03 = a[offseta + (i + 0) + (row + 3) * lda];
        double a11 = a[offseta + (i + 1) + (row + 1) * lda];
        double a12 = a[offseta + (i + 1) + (row + 2) * lda];
        double a13 = a[offseta + (i + 1) + (row + 3) * lda];
        double a22 = a[offseta + (i + 2) + (row + 2) * lda];
        double a23 = a[offseta + (i + 2) + (row + 3) * lda];
        double a33 = a[offseta + (i + 3) + (row + 3) * lda];
        double b0 = b[offsetb + (i + 0) + col * ldb];
        double b1 = b[offsetb + (i + 1) + col * ldb];
        double b2 = b[offsetb + (i + 2) + col * ldb];
        double b3 = b[offsetb + (i + 3) + col * ldb];
        sum0 += b0 * a00 + b1 * a01 + b2 * a02 + b3 * a03;
        sum1 += b0 * a01 + b1 * a11 + b2 * a12 + b3 * a13;
        sum2 += b0 * a02 + b1 * a12 + b2 * a22 + b3 * a23;
        sum3 += b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
        if (beta != 0.0) {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0 + beta * c[offsetc + (row + 0) + col * ldc];
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1 + beta * c[offsetc + (row + 1) + col * ldc];
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2 + beta * c[offsetc + (row + 2) + col * ldc];
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3 + beta * c[offsetc + (row + 3) + col * ldc];
        } else {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0;
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1;
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2;
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3;
        }
      }
      for (; row < m; row += 1) {
        double alphab = alpha * b[offsetb + row + col * ldb];
        double sum = 0.0;
        int i = 0;
        for (; i < row; i += 1) {
          double aval = a[offseta + i + row * lda];
          c[offsetc + i + col * ldc] += alphab * aval;
          sum += b[offsetb + i + col * ldb] * aval;
        }
        sum += b[offsetb + i + col * ldb] * a[offseta + i + row * lda];
        if (beta != 0.0) {
          c[offsetc + row + col * ldc] = alpha * sum + beta * c[offsetc + row + col * ldc];
        } else {
          c[offsetc + row + col * ldc] = alpha * sum;
        }
      }
    }
  }

  protected void dsymmLL(int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    // C := alpha*A*B + beta*C
    f2j.dsymm("L", "L", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  protected void dsymmRU(int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    // C := alpha*B*A + beta*C
    f2j.dsymm("R", "U", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  protected void dsymmRL(int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    // C := alpha*B*A + beta*C
    f2j.dsymm("R", "L", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc) {
    ssymm(side, uplo, m, n, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  public void ssymm(String side, String uplo, int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    if (!lsame("L", side) && !lsame("R", side)) {
      throw illegalArgument("SSYMM", 1);
    }
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSYMM", 2);
    }
    if (m < 0) {
      throw illegalArgument("SSYMM", 3);
    }
    if (n < 0) {
      throw illegalArgument("SSYMM", 4);
    }
    if (lda < Math.max(1, lsame("L", side) ? m : n)) {
      throw illegalArgument("SSYMM", 7);
    }
    if (ldb < Math.max(1, m)) {
      throw illegalArgument("SSYMM", 9);
    }
    if (ldc < Math.max(1, m)) {
      throw illegalArgument("SSYMM", 12);
    }
    if (m == 0 || n == 0 || (alpha == 0.0f && beta == 1.0f)) {
      return;
    }
    if (alpha == 0.0f) {
      // C := beta*C
      int col = 0;
      for (; col < loopBound(n, 4); col += 4) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + (col + 0) * ldc] = beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = beta * c[offsetc + row + (col + 3) * ldc];
        }
      }
      for (; col < n; col += 1) {
        int row = 0;
        for (; row < m; row += 1) {
          c[offsetc + row + col * ldc] = beta * c[offsetc + row + col * ldc];
        }
      }
    } else if (lsame("L", side) && lsame("U", uplo)) {
      ssymmLU(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("L", side) && lsame("L", uplo)) {
      ssymmLL(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("R", side) && lsame("U", uplo)) {
      ssymmRU(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    } else if (lsame("R", side) && lsame("L", uplo)) {
      ssymmRL(m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
    }
  }

  protected void ssymmLU(int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    // C := alpha*A*B + beta*C
    int col = 0;
    for (; col < loopBound(n, 4); col += 4) {
      int row = 0;
      for (; row < loopBound(m, 4); row += 4) {
        float sum00 = 0.0f;
        float sum10 = 0.0f;
        float sum20 = 0.0f;
        float sum30 = 0.0f;
        float sum01 = 0.0f;
        float sum11 = 0.0f;
        float sum21 = 0.0f;
        float sum31 = 0.0f;
        float sum02 = 0.0f;
        float sum12 = 0.0f;
        float sum22 = 0.0f;
        float sum32 = 0.0f;
        float sum03 = 0.0f;
        float sum13 = 0.0f;
        float sum23 = 0.0f;
        float sum33 = 0.0f;
        float alphab00 = alpha * b[offsetb + (row + 0) + (col + 0) * ldb];
        float alphab10 = alpha * b[offsetb + (row + 1) + (col + 0) * ldb];
        float alphab20 = alpha * b[offsetb + (row + 2) + (col + 0) * ldb];
        float alphab30 = alpha * b[offsetb + (row + 3) + (col + 0) * ldb];
        float alphab01 = alpha * b[offsetb + (row + 0) + (col + 1) * ldb];
        float alphab11 = alpha * b[offsetb + (row + 1) + (col + 1) * ldb];
        float alphab21 = alpha * b[offsetb + (row + 2) + (col + 1) * ldb];
        float alphab31 = alpha * b[offsetb + (row + 3) + (col + 1) * ldb];
        float alphab02 = alpha * b[offsetb + (row + 0) + (col + 2) * ldb];
        float alphab12 = alpha * b[offsetb + (row + 1) + (col + 2) * ldb];
        float alphab22 = alpha * b[offsetb + (row + 2) + (col + 2) * ldb];
        float alphab32 = alpha * b[offsetb + (row + 3) + (col + 2) * ldb];
        float alphab03 = alpha * b[offsetb + (row + 0) + (col + 3) * ldb];
        float alphab13 = alpha * b[offsetb + (row + 1) + (col + 3) * ldb];
        float alphab23 = alpha * b[offsetb + (row + 2) + (col + 3) * ldb];
        float alphab33 = alpha * b[offsetb + (row + 3) + (col + 3) * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          float a0 = a[offseta + i + (row + 0) * lda];
          float a1 = a[offseta + i + (row + 1) * lda];
          float a2 = a[offseta + i + (row + 2) * lda];
          float a3 = a[offseta + i + (row + 3) * lda];
          c[offsetc + i + (col + 0) * ldc] += alphab00 * a0
                                           +  alphab10 * a1
                                           +  alphab20 * a2
                                           +  alphab30 * a3;
          c[offsetc + i + (col + 1) * ldc] += alphab01 * a0
                                           +  alphab11 * a1
                                           +  alphab21 * a2
                                           +  alphab31 * a3;
          c[offsetc + i + (col + 2) * ldc] += alphab02 * a0
                                           +  alphab12 * a1
                                           +  alphab22 * a2
                                           +  alphab32 * a3;
          c[offsetc + i + (col + 3) * ldc] += alphab03 * a0
                                           +  alphab13 * a1
                                           +  alphab23 * a2
                                           +  alphab33 * a3;
          float b0 = b[offsetb + i + (col + 0) * ldb];
          float b1 = b[offsetb + i + (col + 1) * ldb];
          float b2 = b[offsetb + i + (col + 2) * ldb];
          float b3 = b[offsetb + i + (col + 3) * ldb];
          sum00 += a0 * b0;
          sum10 += a1 * b0;
          sum20 += a2 * b0;
          sum30 += a3 * b0;
          sum01 += a0 * b1;
          sum11 += a1 * b1;
          sum21 += a2 * b1;
          sum31 += a3 * b1;
          sum02 += a0 * b2;
          sum12 += a1 * b2;
          sum22 += a2 * b2;
          sum32 += a3 * b2;
          sum03 += a0 * b3;
          sum13 += a1 * b3;
          sum23 += a2 * b3;
          sum33 += a3 * b3;
        }
        float a00 = a[offseta + (i + 0) + (row + 0) * lda];
        float a01 = a[offseta + (i + 0) + (row + 1) * lda];
        float a02 = a[offseta + (i + 0) + (row + 2) * lda];
        float a03 = a[offseta + (i + 0) + (row + 3) * lda];
        float a11 = a[offseta + (i + 1) + (row + 1) * lda];
        float a12 = a[offseta + (i + 1) + (row + 2) * lda];
        float a13 = a[offseta + (i + 1) + (row + 3) * lda];
        float a22 = a[offseta + (i + 2) + (row + 2) * lda];
        float a23 = a[offseta + (i + 2) + (row + 3) * lda];
        float a33 = a[offseta + (i + 3) + (row + 3) * lda];
        float b00 = b[offsetb + (i + 0) + (col + 0) * ldb];
        float b10 = b[offsetb + (i + 1) + (col + 0) * ldb];
        float b20 = b[offsetb + (i + 2) + (col + 0) * ldb];
        float b30 = b[offsetb + (i + 3) + (col + 0) * ldb];
        float b01 = b[offsetb + (i + 0) + (col + 1) * ldb];
        float b11 = b[offsetb + (i + 1) + (col + 1) * ldb];
        float b21 = b[offsetb + (i + 2) + (col + 1) * ldb];
        float b31 = b[offsetb + (i + 3) + (col + 1) * ldb];
        float b02 = b[offsetb + (i + 0) + (col + 2) * ldb];
        float b12 = b[offsetb + (i + 1) + (col + 2) * ldb];
        float b22 = b[offsetb + (i + 2) + (col + 2) * ldb];
        float b32 = b[offsetb + (i + 3) + (col + 2) * ldb];
        float b03 = b[offsetb + (i + 0) + (col + 3) * ldb];
        float b13 = b[offsetb + (i + 1) + (col + 3) * ldb];
        float b23 = b[offsetb + (i + 2) + (col + 3) * ldb];
        float b33 = b[offsetb + (i + 3) + (col + 3) * ldb];
        sum00 += a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
        sum10 += a01 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
        sum20 += a02 * b00 + a12 * b10 + a22 * b20 + a23 * b30;
        sum30 += a03 * b00 + a13 * b10 + a23 * b20 + a33 * b30;
        sum01 += a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
        sum11 += a01 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
        sum21 += a02 * b01 + a12 * b11 + a22 * b21 + a23 * b31;
        sum31 += a03 * b01 + a13 * b11 + a23 * b21 + a33 * b31;
        sum02 += a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
        sum12 += a01 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
        sum22 += a02 * b02 + a12 * b12 + a22 * b22 + a23 * b32;
        sum32 += a03 * b02 + a13 * b12 + a23 * b22 + a33 * b32;
        sum03 += a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
        sum13 += a01 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
        sum23 += a02 * b03 + a12 * b13 + a22 * b23 + a23 * b33;
        sum33 += a03 * b03 + a13 * b13 + a23 * b23 + a33 * b33;
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00 + beta * c[offsetc + (row + 0) + (col + 0) * ldc];
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum10 + beta * c[offsetc + (row + 1) + (col + 0) * ldc];
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum20 + beta * c[offsetc + (row + 2) + (col + 0) * ldc];
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum30 + beta * c[offsetc + (row + 3) + (col + 0) * ldc];
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum01 + beta * c[offsetc + (row + 0) + (col + 1) * ldc];
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11 + beta * c[offsetc + (row + 1) + (col + 1) * ldc];
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum21 + beta * c[offsetc + (row + 2) + (col + 1) * ldc];
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum31 + beta * c[offsetc + (row + 3) + (col + 1) * ldc];
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum02 + beta * c[offsetc + (row + 0) + (col + 2) * ldc];
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum12 + beta * c[offsetc + (row + 1) + (col + 2) * ldc];
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22 + beta * c[offsetc + (row + 2) + (col + 2) * ldc];
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum32 + beta * c[offsetc + (row + 3) + (col + 2) * ldc];
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum03 + beta * c[offsetc + (row + 0) + (col + 3) * ldc];
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum13 + beta * c[offsetc + (row + 1) + (col + 3) * ldc];
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum23 + beta * c[offsetc + (row + 2) + (col + 3) * ldc];
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33 + beta * c[offsetc + (row + 3) + (col + 3) * ldc];
        } else {
          c[offsetc + (row + 0) + (col + 0) * ldc] = alpha * sum00;
          c[offsetc + (row + 1) + (col + 0) * ldc] = alpha * sum10;
          c[offsetc + (row + 2) + (col + 0) * ldc] = alpha * sum20;
          c[offsetc + (row + 3) + (col + 0) * ldc] = alpha * sum30;
          c[offsetc + (row + 0) + (col + 1) * ldc] = alpha * sum01;
          c[offsetc + (row + 1) + (col + 1) * ldc] = alpha * sum11;
          c[offsetc + (row + 2) + (col + 1) * ldc] = alpha * sum21;
          c[offsetc + (row + 3) + (col + 1) * ldc] = alpha * sum31;
          c[offsetc + (row + 0) + (col + 2) * ldc] = alpha * sum02;
          c[offsetc + (row + 1) + (col + 2) * ldc] = alpha * sum12;
          c[offsetc + (row + 2) + (col + 2) * ldc] = alpha * sum22;
          c[offsetc + (row + 3) + (col + 2) * ldc] = alpha * sum32;
          c[offsetc + (row + 0) + (col + 3) * ldc] = alpha * sum03;
          c[offsetc + (row + 1) + (col + 3) * ldc] = alpha * sum13;
          c[offsetc + (row + 2) + (col + 3) * ldc] = alpha * sum23;
          c[offsetc + (row + 3) + (col + 3) * ldc] = alpha * sum33;
        }
      }
      for (; row < m; row += 1) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float alphab0 = alpha * b[offsetb + row + (col + 0) * ldb];
        float alphab1 = alpha * b[offsetb + row + (col + 1) * ldb];
        float alphab2 = alpha * b[offsetb + row + (col + 2) * ldb];
        float alphab3 = alpha * b[offsetb + row + (col + 3) * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          float a0 = a[offseta + i + row * lda];
          c[offsetc + i + (col + 0) * ldc] += alphab0 * a0;
          c[offsetc + i + (col + 1) * ldc] += alphab1 * a0;
          c[offsetc + i + (col + 2) * ldc] += alphab2 * a0;
          c[offsetc + i + (col + 3) * ldc] += alphab3 * a0;
          sum0 += b[offsetb + i + (col + 0) * ldb] * a0;
          sum1 += b[offsetb + i + (col + 1) * ldb] * a0;
          sum2 += b[offsetb + i + (col + 2) * ldb] * a0;
          sum3 += b[offsetb + i + (col + 3) * ldb] * a0;
        }
        float a0 = a[offseta + i + row * lda];
        sum0 += b[offsetb + i + (col + 0) * ldb] * a0;
        sum1 += b[offsetb + i + (col + 1) * ldb] * a0;
        sum2 += b[offsetb + i + (col + 2) * ldb] * a0;
        sum3 += b[offsetb + i + (col + 3) * ldb] * a0;
        if (beta != 0.0f) {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0 + beta * c[offsetc + row + (col + 0) * ldc];
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1 + beta * c[offsetc + row + (col + 1) * ldc];
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2 + beta * c[offsetc + row + (col + 2) * ldc];
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3 + beta * c[offsetc + row + (col + 3) * ldc];
        } else {
          c[offsetc + row + (col + 0) * ldc] = alpha * sum0;
          c[offsetc + row + (col + 1) * ldc] = alpha * sum1;
          c[offsetc + row + (col + 2) * ldc] = alpha * sum2;
          c[offsetc + row + (col + 3) * ldc] = alpha * sum3;
        }
      }
    }
    for (; col < n; col += 1) {
      int row = 0;
      for (; row < loopBound(m, 4); row += 4) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float alphab0 = alpha * b[offsetb + (row + 0) + col * ldb];
        float alphab1 = alpha * b[offsetb + (row + 1) + col * ldb];
        float alphab2 = alpha * b[offsetb + (row + 2) + col * ldb];
        float alphab3 = alpha * b[offsetb + (row + 3) + col * ldb];
        int i = 0;
        for (; i < row; i += 1) {
          float a0 = a[offseta + i + (row + 0) * lda];
          float a1 = a[offseta + i + (row + 1) * lda];
          float a2 = a[offseta + i + (row + 2) * lda];
          float a3 = a[offseta + i + (row + 3) * lda];
          c[offsetc + i + col * ldc] += alphab0 * a0
                                     +  alphab1 * a1
                                     +  alphab2 * a2
                                     +  alphab3 * a3;
          float b0 = b[offsetb + i + col * ldb];
          sum0 += b0 * a0;
          sum1 += b0 * a1;
          sum2 += b0 * a2;
          sum3 += b0 * a3;
        }
        float a00 = a[offseta + (i + 0) + (row + 0) * lda];
        float a01 = a[offseta + (i + 0) + (row + 1) * lda];
        float a02 = a[offseta + (i + 0) + (row + 2) * lda];
        float a03 = a[offseta + (i + 0) + (row + 3) * lda];
        float a11 = a[offseta + (i + 1) + (row + 1) * lda];
        float a12 = a[offseta + (i + 1) + (row + 2) * lda];
        float a13 = a[offseta + (i + 1) + (row + 3) * lda];
        float a22 = a[offseta + (i + 2) + (row + 2) * lda];
        float a23 = a[offseta + (i + 2) + (row + 3) * lda];
        float a33 = a[offseta + (i + 3) + (row + 3) * lda];
        float b0 = b[offsetb + (i + 0) + col * ldb];
        float b1 = b[offsetb + (i + 1) + col * ldb];
        float b2 = b[offsetb + (i + 2) + col * ldb];
        float b3 = b[offsetb + (i + 3) + col * ldb];
        sum0 += b0 * a00 + b1 * a01 + b2 * a02 + b3 * a03;
        sum1 += b0 * a01 + b1 * a11 + b2 * a12 + b3 * a13;
        sum2 += b0 * a02 + b1 * a12 + b2 * a22 + b3 * a23;
        sum3 += b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
        if (beta != 0.0f) {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0 + beta * c[offsetc + (row + 0) + col * ldc];
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1 + beta * c[offsetc + (row + 1) + col * ldc];
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2 + beta * c[offsetc + (row + 2) + col * ldc];
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3 + beta * c[offsetc + (row + 3) + col * ldc];
        } else {
          c[offsetc + (row + 0) + col * ldc] = alpha * sum0;
          c[offsetc + (row + 1) + col * ldc] = alpha * sum1;
          c[offsetc + (row + 2) + col * ldc] = alpha * sum2;
          c[offsetc + (row + 3) + col * ldc] = alpha * sum3;
        }
      }
      for (; row < m; row += 1) {
        float alphab = alpha * b[offsetb + row + col * ldb];
        float sum = 0.0f;
        int i = 0;
        for (; i < row; i += 1) {
          float aval = a[offseta + i + row * lda];
          c[offsetc + i + col * ldc] += alphab * aval;
          sum += b[offsetb + i + col * ldb] * aval;
        }
        sum += b[offsetb + i + col * ldb] * a[offseta + i + row * lda];
        if (beta != 0.0f) {
          c[offsetc + row + col * ldc] = alpha * sum + beta * c[offsetc + row + col * ldc];
        } else {
          c[offsetc + row + col * ldc] = alpha * sum;
        }
      }
    }
  }

  protected void ssymmLL(int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    // C := alpha*A*B + beta*C
    f2j.ssymm("L", "L", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  protected void ssymmRU(int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    // C := alpha*B*A + beta*C
    f2j.ssymm("R", "U", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  protected void ssymmRL(int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    // C := alpha*B*A + beta*C
    f2j.ssymm("R", "L", m, n, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  public void dsymv(String uplo, int n, double alpha, double[] a, int lda, double[] x, int incx, double beta, double[] y, int incy) {
    dsymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void dsymv(String uplo, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSYMV", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSYMV", 2);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("DSYMV", 5);
    }
    if (incx == 0) {
      throw illegalArgument("DSYMV", 7);
    }
    if (incy == 0) {
      throw illegalArgument("DSYMV", 10);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0;
        }
      }
    } else if (lsame("U", uplo)) {
      dsymvU(n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("L", uplo)) {
      dsymvL(n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void dsymvU(int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double sumiy0 = 0.0;
      double sumiy1 = 0.0;
      double sumiy2 = 0.0;
      double sumiy3 = 0.0;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix0 * a[offseta + row + (col + 0) * lda]
                        +  alphaxix1 * a[offseta + row + (col + 1) * lda]
                        +  alphaxix2 * a[offseta + row + (col + 2) * lda]
                        +  alphaxix3 * a[offseta + row + (col + 3) * lda];
        double xjx = x[offsetx + jx];
        sumiy0 += xjx * a[offseta + row + (col + 0) * lda];
        sumiy1 += xjx * a[offseta + row + (col + 1) * lda];
        sumiy2 += xjx * a[offseta + row + (col + 2) * lda];
        sumiy3 += xjx * a[offseta + row + (col + 3) * lda];
      }
      double a00 = a[offseta + (row + 0) + (col + 0) * lda];
      double a01 = a[offseta + (row + 0) + (col + 1) * lda];
      double a02 = a[offseta + (row + 0) + (col + 2) * lda];
      double a03 = a[offseta + (row + 0) + (col + 3) * lda];
      double a11 = a[offseta + (row + 1) + (col + 1) * lda];
      double a12 = a[offseta + (row + 1) + (col + 2) * lda];
      double a13 = a[offseta + (row + 1) + (col + 3) * lda];
      double a22 = a[offseta + (row + 2) + (col + 2) * lda];
      double a23 = a[offseta + (row + 2) + (col + 3) * lda];
      double a33 = a[offseta + (row + 3) + (col + 3) * lda];
      double xjx0 = x[offsetx + jx + incx * 0];
      double xjx1 = x[offsetx + jx + incx * 1];
      double xjx2 = x[offsetx + jx + incx * 2];
      double xjx3 = x[offsetx + jx + incx * 3];
      sumiy0 += xjx0 * a00
             +  xjx1 * a01
             +  xjx2 * a02
             +  xjx3 * a03;
      sumiy1 += xjx0 * a01
             +  xjx1 * a11
             +  xjx2 * a12
             +  xjx3 * a13;
      sumiy2 += xjx0 * a02
             +  xjx1 * a12
             +  xjx2 * a22
             +  xjx3 * a23;
      sumiy3 += xjx0 * a03
             +  xjx1 * a13
             +  xjx2 * a23
             +  xjx3 * a33;
      if (beta != 0.0) {
        y[offsety + iy + incy * 0] = alpha * sumiy0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sumiy1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sumiy2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sumiy3 + beta * y[offsety + iy + incy * 3];
      } else {
        y[offsety + iy + incy * 0] = alpha * sumiy0;
        y[offsety + iy + incy * 1] = alpha * sumiy1;
        y[offsety + iy + incy * 2] = alpha * sumiy2;
        y[offsety + iy + incy * 3] = alpha * sumiy3;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double sumiy = 0.0;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * lda];
        sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      }
      sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      if (beta != 0.0) {
        y[offsety + iy] = alpha * sumiy + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sumiy;
      }
    }
  }

  protected void dsymvL(int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy) {
    // y = beta * y
    if (beta != 1.0) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0;
        }
      }
    }
    // y += alpha * A * x 
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double sumiy0 = 0.0;
      double sumiy1 = 0.0;
      double sumiy2 = 0.0;
      double sumiy3 = 0.0;
      double a00 = a[offseta + /*row=*/(col + 0) + (col + 0) * lda];
      double a10 = a[offseta + /*row=*/(col + 1) + (col + 0) * lda];
      double a11 = a[offseta + /*row=*/(col + 1) + (col + 1) * lda];
      double a20 = a[offseta + /*row=*/(col + 2) + (col + 0) * lda];
      double a21 = a[offseta + /*row=*/(col + 2) + (col + 1) * lda];
      double a22 = a[offseta + /*row=*/(col + 2) + (col + 2) * lda];
      double a30 = a[offseta + /*row=*/(col + 3) + (col + 0) * lda];
      double a31 = a[offseta + /*row=*/(col + 3) + (col + 1) * lda];
      double a32 = a[offseta + /*row=*/(col + 3) + (col + 2) * lda];
      double a33 = a[offseta + /*row=*/(col + 3) + (col + 3) * lda];
      double x0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      double x1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      double x2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      double x3 = x[offsetx + (incx < 0 ? (n - (col + 3) - 1) * -incx : (col + 3) * incx)];
      sumiy0 += x0 * a00
             +  x1 * a10
             +  x2 * a20
             +  x3 * a30;
      sumiy1 += x0 * a10
             +  x1 * a11
             +  x2 * a21
             +  x3 * a31;
      sumiy2 += x0 * a20
             +  x1 * a21
             +  x2 * a22
             +  x3 * a32;
      sumiy3 += x0 * a30
             +  x1 * a31
             +  x2 * a32
             +  x3 * a33;
      int row = col + 4, jx = incx < 0 ? (n - (col + 4) - 1) * -incx : (col + 4) * incx, jy = incy < 0 ? (n - (col + 4) - 1) * -incy : (col + 4) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        double a0 = a[offseta + row + (col + 0) * lda];
        double a1 = a[offseta + row + (col + 1) * lda];
        double a2 = a[offseta + row + (col + 2) * lda];
        double a3 = a[offseta + row + (col + 3) * lda];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        double xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      y[offsety + iy + incy * 0] += alpha * sumiy0;
      y[offsety + iy + incy * 1] += alpha * sumiy1;
      y[offsety + iy + incy * 2] += alpha * sumiy2;
      y[offsety + iy + incy * 3] += alpha * sumiy3;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double sumiy = 0.0;
      sumiy += x[offsetx + (incx < 0 ? (n - col - 1) * -incx : col * incx)] * a[offseta + /*row=*/col + col * lda];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * lda];
        sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      }
      y[offsety + iy] += alpha * sumiy;
    }
  }

  public void ssymv(String uplo, int n, float alpha, float[] a, int lda, float[] x, int incx, float beta, float[] y, int incy) {
    ssymv(uplo, n, alpha, a, 0, lda, x, 0, incx, beta, y, 0, incy);
  }

  public void ssymv(String uplo, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSYMV", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSYMV", 2);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("SSYMV", 5);
    }
    if (incx == 0) {
      throw illegalArgument("SSYMV", 7);
    }
    if (incy == 0) {
      throw illegalArgument("SSYMV", 10);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0f) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0f;
        }
      }
    } else if (lsame("U", uplo)) {
      ssymvU(n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    } else if (lsame("L", uplo)) {
      ssymvL(n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
    }
  }

  protected void ssymvU(int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float sumiy0 = 0.0f;
      float sumiy1 = 0.0f;
      float sumiy2 = 0.0f;
      float sumiy3 = 0.0f;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix0 * a[offseta + row + (col + 0) * lda]
                        +  alphaxix1 * a[offseta + row + (col + 1) * lda]
                        +  alphaxix2 * a[offseta + row + (col + 2) * lda]
                        +  alphaxix3 * a[offseta + row + (col + 3) * lda];
        float xjx = x[offsetx + jx];
        sumiy0 += xjx * a[offseta + row + (col + 0) * lda];
        sumiy1 += xjx * a[offseta + row + (col + 1) * lda];
        sumiy2 += xjx * a[offseta + row + (col + 2) * lda];
        sumiy3 += xjx * a[offseta + row + (col + 3) * lda];
      }
      float a00 = a[offseta + (row + 0) + (col + 0) * lda];
      float a01 = a[offseta + (row + 0) + (col + 1) * lda];
      float a02 = a[offseta + (row + 0) + (col + 2) * lda];
      float a03 = a[offseta + (row + 0) + (col + 3) * lda];
      float a11 = a[offseta + (row + 1) + (col + 1) * lda];
      float a12 = a[offseta + (row + 1) + (col + 2) * lda];
      float a13 = a[offseta + (row + 1) + (col + 3) * lda];
      float a22 = a[offseta + (row + 2) + (col + 2) * lda];
      float a23 = a[offseta + (row + 2) + (col + 3) * lda];
      float a33 = a[offseta + (row + 3) + (col + 3) * lda];
      float xjx0 = x[offsetx + jx + incx * 0];
      float xjx1 = x[offsetx + jx + incx * 1];
      float xjx2 = x[offsetx + jx + incx * 2];
      float xjx3 = x[offsetx + jx + incx * 3];
      sumiy0 += xjx0 * a00
             +  xjx1 * a01
             +  xjx2 * a02
             +  xjx3 * a03;
      sumiy1 += xjx0 * a01
             +  xjx1 * a11
             +  xjx2 * a12
             +  xjx3 * a13;
      sumiy2 += xjx0 * a02
             +  xjx1 * a12
             +  xjx2 * a22
             +  xjx3 * a23;
      sumiy3 += xjx0 * a03
             +  xjx1 * a13
             +  xjx2 * a23
             +  xjx3 * a33;
      if (beta != 0.0f) {
        y[offsety + iy + incy * 0] = alpha * sumiy0 + beta * y[offsety + iy + incy * 0];
        y[offsety + iy + incy * 1] = alpha * sumiy1 + beta * y[offsety + iy + incy * 1];
        y[offsety + iy + incy * 2] = alpha * sumiy2 + beta * y[offsety + iy + incy * 2];
        y[offsety + iy + incy * 3] = alpha * sumiy3 + beta * y[offsety + iy + incy * 3];
      } else {
        y[offsety + iy + incy * 0] = alpha * sumiy0;
        y[offsety + iy + incy * 1] = alpha * sumiy1;
        y[offsety + iy + incy * 2] = alpha * sumiy2;
        y[offsety + iy + incy * 3] = alpha * sumiy3;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float sumiy = 0.0f;
      int row = 0, jx = incx < 0 ? (col - 1) * -incx : 0, jy = incy < 0 ? (col - 1) * -incy : 0;
      for (; row < col; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * lda];
        sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      }
      sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      if (beta != 0.0f) {
        y[offsety + iy] = alpha * sumiy + beta * y[offsety + iy];
      } else {
        y[offsety + iy] = alpha * sumiy;
      }
    }
  }

  protected void ssymvL(int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy) {
    // y = beta * y
    if (beta != 1.0f) {
      for (int i = 0, iy = incy < 0 ? (n - 1) * -incy : 0; i < n; i += 1, iy += incy) {
        if (beta != 0.0f) {
          y[offsety + iy] = beta * y[offsety + iy];
        } else {
          y[offsety + iy] = 0.0f;
        }
      }
    }
    // y += alpha * A * x 
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float sumiy0 = 0.0f;
      float sumiy1 = 0.0f;
      float sumiy2 = 0.0f;
      float sumiy3 = 0.0f;
      float a00 = a[offseta + /*row=*/(col + 0) + (col + 0) * lda];
      float a10 = a[offseta + /*row=*/(col + 1) + (col + 0) * lda];
      float a11 = a[offseta + /*row=*/(col + 1) + (col + 1) * lda];
      float a20 = a[offseta + /*row=*/(col + 2) + (col + 0) * lda];
      float a21 = a[offseta + /*row=*/(col + 2) + (col + 1) * lda];
      float a22 = a[offseta + /*row=*/(col + 2) + (col + 2) * lda];
      float a30 = a[offseta + /*row=*/(col + 3) + (col + 0) * lda];
      float a31 = a[offseta + /*row=*/(col + 3) + (col + 1) * lda];
      float a32 = a[offseta + /*row=*/(col + 3) + (col + 2) * lda];
      float a33 = a[offseta + /*row=*/(col + 3) + (col + 3) * lda];
      float x0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      float x1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      float x2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      float x3 = x[offsetx + (incx < 0 ? (n - (col + 3) - 1) * -incx : (col + 3) * incx)];
      sumiy0 += x0 * a00
             +  x1 * a10
             +  x2 * a20
             +  x3 * a30;
      sumiy1 += x0 * a10
             +  x1 * a11
             +  x2 * a21
             +  x3 * a31;
      sumiy2 += x0 * a20
             +  x1 * a21
             +  x2 * a22
             +  x3 * a32;
      sumiy3 += x0 * a30
             +  x1 * a31
             +  x2 * a32
             +  x3 * a33;
      int row = col + 4, jx = incx < 0 ? (n - (col + 4) - 1) * -incx : (col + 4) * incx, jy = incy < 0 ? (n - (col + 4) - 1) * -incy : (col + 4) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        float a0 = a[offseta + row + (col + 0) * lda];
        float a1 = a[offseta + row + (col + 1) * lda];
        float a2 = a[offseta + row + (col + 2) * lda];
        float a3 = a[offseta + row + (col + 3) * lda];
        y[offsety + jy] += alphaxix0 * a0
                        +  alphaxix1 * a1
                        +  alphaxix2 * a2
                        +  alphaxix3 * a3;
        float xjx = x[offsetx + jx];
        sumiy0 += xjx * a0;
        sumiy1 += xjx * a1;
        sumiy2 += xjx * a2;
        sumiy3 += xjx * a3;
      }
      y[offsety + iy + incy * 0] += alpha * sumiy0;
      y[offsety + iy + incy * 1] += alpha * sumiy1;
      y[offsety + iy + incy * 2] += alpha * sumiy2;
      y[offsety + iy + incy * 3] += alpha * sumiy3;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float sumiy = 0.0f;
      sumiy += x[offsetx + (incx < 0 ? (n - col - 1) * -incx : col * incx)] * a[offseta + /*row=*/col + col * lda];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy;
      for (; row < n; row += 1, jx += incx, jy += incy) {
        y[offsety + jy] += alphaxix * a[offseta + row + col * lda];
        sumiy += x[offsetx + jx] * a[offseta + row + col * lda];
      }
      y[offsety + iy] += alpha * sumiy;
    }
  }

  public void dsyr(String uplo, int n, double alpha, double[] x, int incx, double[] a, int lda) {
    dsyr(uplo, n, alpha, x, 0, incx, a, 0, lda);
  }

  // a += alpha * x * x.t
  public void dsyr(String uplo, int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta, int lda) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSYR", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSYR", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DSYR", 5);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("DSYR", 7);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      // do nothing
    } else if (lsame("U", uplo)) {
      dsyrU(n, alpha, x, offsetx, incx, a, offseta, lda);
    } else if (lsame("L", uplo)) {
      dsyrL(n, alpha, x, offsetx, incx, a, offseta, lda);
    }
  }

  protected void dsyrU(int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * xjx;
      }
      double xjx0 = x[offsetx + jx + incx * 0];
      a[offseta + (row + 0) + (col + 1) * lda] += alphaxix1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * lda] += alphaxix2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * lda] += alphaxix3 * xjx0;
      double xjx1 = x[offsetx + jx + incx * 1];
      a[offseta + (row + 1) + (col + 2) * lda] += alphaxix2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * lda] += alphaxix3 * xjx1;
      double xjx2 = x[offsetx + jx + incx * 2];
      a[offseta + (row + 2) + (col + 3) * lda] += alphaxix3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx) {
      double alphaxix = alpha * x[offsetx + ix];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        a[offseta + row + col * lda] += alphaxix * x[offsetx + jx];
      }
    }
  }

  protected void dsyrL(int n, double alpha, double[] x, int offsetx, int incx, double[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * lda] += alphaxix0 * xjx0;
      double xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * lda] += alphaxix0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * lda] += alphaxix1 * xjx1;
      double xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * lda] += alphaxix0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * lda] += alphaxix1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * lda] += alphaxix2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx;
      for (; row < n; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx) {
      double alphaxix = alpha * x[offsetx + ix];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx;
      for (; row < n; row += 1, jx += incx) {
        a[offseta + row + col * lda] += alphaxix * x[offsetx + jx];
      }
    }
  }

  public void ssyr(String uplo, int n, float alpha, float[] x, int incx, float[] a, int lda) {
    ssyr(uplo, n, alpha, x, 0, incx, a, 0, lda);
  }

  public void ssyr(String uplo, int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta, int lda) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSYR", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSYR", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SSYR", 5);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("SSYR", 7);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      // do nothing
    } else if (lsame("U", uplo)) {
      ssyrU(n, alpha, x, offsetx, incx, a, offseta, lda);
    } else if (lsame("L", uplo)) {
      ssyrL(n, alpha, x, offsetx, incx, a, offseta, lda);
    }
  }

  protected void ssyrU(int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * xjx;
      }
      float xjx0 = x[offsetx + jx + incx * 0];
      a[offseta + (row + 0) + (col + 1) * lda] += alphaxix1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * lda] += alphaxix2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * lda] += alphaxix3 * xjx0;
      float xjx1 = x[offsetx + jx + incx * 1];
      a[offseta + (row + 1) + (col + 2) * lda] += alphaxix2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * lda] += alphaxix3 * xjx1;
      float xjx2 = x[offsetx + jx + incx * 2];
      a[offseta + (row + 2) + (col + 3) * lda] += alphaxix3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx) {
      float alphaxix = alpha * x[offsetx + ix];
      int row = 0, jx = incx < 0 ? col * -incx : 0;
      for (; row < col + 1; row += 1, jx += incx) {
        a[offseta + row + col * lda] += alphaxix * x[offsetx + jx];
      }
    }
  }

  protected void ssyrL(int n, float alpha, float[] x, int offsetx, int incx, float[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * lda] += alphaxix0 * xjx0;
      float xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * lda] += alphaxix0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * lda] += alphaxix1 * xjx1;
      float xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * lda] += alphaxix0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * lda] += alphaxix1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * lda] += alphaxix2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx;
      for (; row < n; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx) {
      float alphaxix = alpha * x[offsetx + ix];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx;
      for (; row < n; row += 1, jx += incx) {
        a[offseta + row + col * lda] += alphaxix * x[offsetx + jx];
      }
    }
  }

  public void dsyr2(String uplo, int n, double alpha, double[] x, int incx, double[] y, int incy, double[] a, int lda) {
    dsyr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
  }

  public void dsyr2(String uplo, int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta, int lda) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("DSYR2", 1);
    }
    if (n < 0) {
      throw illegalArgument("DSYR2", 2);
    }
    if (incx == 0) {
      throw illegalArgument("DSYR2", 5);
    }
    if (incy == 0) {
      throw illegalArgument("DSYR2", 7);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("DSYR2", 9);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0) {
      // do nothing
    } else if (lsame("U", uplo)) {
      dsyr2U(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta, lda);
    } else if (lsame("L", uplo)) {
      dsyr2L(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta, lda);
    }
  }

  protected void dsyr2U(int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double alphayiy0 = alpha * y[offsety + iy + incy * 0];
      double alphayiy1 = alpha * y[offsety + iy + incy * 1];
      double alphayiy2 = alpha * y[offsety + iy + incy * 2];
      double alphayiy3 = alpha * y[offsety + iy + incy * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        double xjx = x[offsetx + jx];
        double yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
      double xjx0 = x[offsetx + jx + incx * 0];
      double yjy0 = y[offsety + jy + incy * 0];
      a[offseta + (row + 0) + (col + 1) * lda] += alphaxix1 * yjy0 + alphayiy1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * lda] += alphaxix2 * yjy0 + alphayiy2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * lda] += alphaxix3 * yjy0 + alphayiy3 * xjx0;
      double xjx1 = x[offsetx + jx + incx * 1];
      double yjy1 = y[offsety + jy + incy * 1];
      a[offseta + (row + 1) + (col + 2) * lda] += alphaxix2 * yjy1 + alphayiy2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * lda] += alphaxix3 * yjy1 + alphayiy3 * xjx1;
      double xjx2 = x[offsetx + jx + incx * 2];
      double yjy2 = y[offsety + jy + incy * 2];
      a[offseta + (row + 2) + (col + 3) * lda] += alphaxix3 * yjy2 + alphayiy3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double alphayiy = alpha * y[offsety + iy];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * lda] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  protected void dsyr2L(int n, double alpha, double[] x, int offsetx, int incx, double[] y, int offsety, int incy, double[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      double alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      double alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      double alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      double alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      double alphayiy0 = alpha * y[offsety + iy + incy * 0];
      double alphayiy1 = alpha * y[offsety + iy + incy * 1];
      double alphayiy2 = alpha * y[offsety + iy + incy * 2];
      double alphayiy3 = alpha * y[offsety + iy + incy * 3];
      double xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      double yjy0 = y[offsety + (incy < 0 ? (n - (col + 0) - 1) * -incy : (col + 0) * incy)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * lda] += alphaxix0 * yjy0 + alphayiy0 * xjx0;
      double xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      double yjy1 = y[offsety + (incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * lda] += alphaxix0 * yjy1 + alphayiy0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * lda] += alphaxix1 * yjy1 + alphayiy1 * xjx1;
      double xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      double yjy2 = y[offsety + (incy < 0 ? (n - (col + 2) - 1) * -incy : (col + 2) * incy)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * lda] += alphaxix0 * yjy2 + alphayiy0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * lda] += alphaxix1 * yjy2 + alphayiy1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * lda] += alphaxix2 * yjy2 + alphayiy2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < n; row += 1, jx += incx) {
        double xjx = x[offsetx + jx];
        double yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      double alphaxix = alpha * x[offsetx + ix];
      double alphayiy = alpha * y[offsety + iy];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * lda] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  public void ssyr2(String uplo, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] a, int lda) {
    ssyr2(uplo, n, alpha, x, 0, incx, y, 0, incy, a, 0, lda);
  }

  public void ssyr2(String uplo, int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta, int lda) {
    if (!lsame("U", uplo) && !lsame("L", uplo)) {
      throw illegalArgument("SSYR2", 1);
    }
    if (n < 0) {
      throw illegalArgument("SSYR2", 2);
    }
    if (incx == 0) {
      throw illegalArgument("SSYR2", 5);
    }
    if (incy == 0) {
      throw illegalArgument("SSYR2", 7);
    }
    if (lda < Math.max(1, n)) {
      throw illegalArgument("SSYR2", 9);
    }
    if (n == 0) {
      return;
    }
    if (alpha == 0.0f) {
      // do nothing
    } else if (lsame("U", uplo)) {
      ssyr2U(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta, lda);
    } else if (lsame("L", uplo)) {
      ssyr2L(n, alpha, x, offsetx, incx, y, offsety, incy, a, offseta, lda);
    }
  }

  protected void ssyr2U(int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float alphayiy0 = alpha * y[offsety + iy + incy * 0];
      float alphayiy1 = alpha * y[offsety + iy + incy * 1];
      float alphayiy2 = alpha * y[offsety + iy + incy * 2];
      float alphayiy3 = alpha * y[offsety + iy + incy * 3];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        float xjx = x[offsetx + jx];
        float yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
      float xjx0 = x[offsetx + jx + incx * 0];
      float yjy0 = y[offsety + jy + incy * 0];
      a[offseta + (row + 0) + (col + 1) * lda] += alphaxix1 * yjy0 + alphayiy1 * xjx0;
      a[offseta + (row + 0) + (col + 2) * lda] += alphaxix2 * yjy0 + alphayiy2 * xjx0;
      a[offseta + (row + 0) + (col + 3) * lda] += alphaxix3 * yjy0 + alphayiy3 * xjx0;
      float xjx1 = x[offsetx + jx + incx * 1];
      float yjy1 = y[offsety + jy + incy * 1];
      a[offseta + (row + 1) + (col + 2) * lda] += alphaxix2 * yjy1 + alphayiy2 * xjx1;
      a[offseta + (row + 1) + (col + 3) * lda] += alphaxix3 * yjy1 + alphayiy3 * xjx1;
      float xjx2 = x[offsetx + jx + incx * 2];
      float yjy2 = y[offsety + jy + incy * 2];
      a[offseta + (row + 2) + (col + 3) * lda] += alphaxix3 * yjy2 + alphayiy3 * xjx2;
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float alphayiy = alpha * y[offsety + iy];
      int row = 0, jx = incx < 0 ? col * -incx : 0, jy = incy < 0 ? (n - 1) * -incy : 0;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * lda] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  protected void ssyr2L(int n, float alpha, float[] x, int offsetx, int incx, float[] y, int offsety, int incy, float[] a, int offseta, int lda) {
    int col = 0, ix = incx < 0 ? (n - 1) * -incx : 0, iy = incy < 0 ? (n - 1) * -incy : 0;
    for (; col < loopBound(n, 4); col += 4, ix += incx * 4, iy += incy * 4) {
      float alphaxix0 = alpha * x[offsetx + ix + incx * 0];
      float alphaxix1 = alpha * x[offsetx + ix + incx * 1];
      float alphaxix2 = alpha * x[offsetx + ix + incx * 2];
      float alphaxix3 = alpha * x[offsetx + ix + incx * 3];
      float alphayiy0 = alpha * y[offsety + iy + incy * 0];
      float alphayiy1 = alpha * y[offsety + iy + incy * 1];
      float alphayiy2 = alpha * y[offsety + iy + incy * 2];
      float alphayiy3 = alpha * y[offsety + iy + incy * 3];
      float xjx0 = x[offsetx + (incx < 0 ? (n - (col + 0) - 1) * -incx : (col + 0) * incx)];
      float yjy0 = y[offsety + (incy < 0 ? (n - (col + 0) - 1) * -incy : (col + 0) * incy)];
      a[offseta + /*row=*/(col + 0) + (col + 0) * lda] += alphaxix0 * yjy0 + alphayiy0 * xjx0;
      float xjx1 = x[offsetx + (incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx)];
      float yjy1 = y[offsety + (incy < 0 ? (n - (col + 1) - 1) * -incy : (col + 1) * incy)];
      a[offseta + /*row=*/(col + 1) + (col + 0) * lda] += alphaxix0 * yjy1 + alphayiy0 * xjx1;
      a[offseta + /*row=*/(col + 1) + (col + 1) * lda] += alphaxix1 * yjy1 + alphayiy1 * xjx1;
      float xjx2 = x[offsetx + (incx < 0 ? (n - (col + 2) - 1) * -incx : (col + 2) * incx)];
      float yjy2 = y[offsety + (incy < 0 ? (n - (col + 2) - 1) * -incy : (col + 2) * incy)];
      a[offseta + /*row=*/(col + 2) + (col + 0) * lda] += alphaxix0 * yjy2 + alphayiy0 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 1) * lda] += alphaxix1 * yjy2 + alphayiy1 * xjx2;
      a[offseta + /*row=*/(col + 2) + (col + 2) * lda] += alphaxix2 * yjy2 + alphayiy2 * xjx2;
      int row = col + (4 - 1), jx = incx < 0 ? (n - (col + (4 - 1)) - 1) * -incx : (col + (4 - 1)) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < n; row += 1, jx += incx) {
        float xjx = x[offsetx + jx];
        float yjy = y[offsety + jy];
        a[offseta + row + (col + 0) * lda] += alphaxix0 * yjy + alphayiy0 * xjx;
        a[offseta + row + (col + 1) * lda] += alphaxix1 * yjy + alphayiy1 * xjx;
        a[offseta + row + (col + 2) * lda] += alphaxix2 * yjy + alphayiy2 * xjx;
        a[offseta + row + (col + 3) * lda] += alphaxix3 * yjy + alphayiy3 * xjx;
      }
    }
    for (; col < n; col += 1, ix += incx, iy += incy) {
      float alphaxix = alpha * x[offsetx + ix];
      float alphayiy = alpha * y[offsety + iy];
      int row = col + 1, jx = incx < 0 ? (n - (col + 1) - 1) * -incx : (col + 1) * incx, jy = incy < 0 ? (n - (col + (4 - 1)) - 1) * -incy : (col + (4 - 1)) * incy;
      for (; row < col + 1; row += 1, jx += incx, jy += incy) {
        a[offseta + row + col * lda] += alphaxix * y[offsety + jy] + alphayiy * x[offsetx + jx];
      }
    }
  }

  public void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int lda, double[] b, int ldb, double beta, double[] c, int ldc) {
    dsyr2k(uplo, trans, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  public void dsyr2k(String uplo, String trans, int n, int k, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc) {
    f2j.dsyr2k(uplo, trans, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  public void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc) {
    ssyr2k(uplo, trans, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
  }

  public void ssyr2k(String uplo, String trans, int n, int k, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc) {
    f2j.ssyr2k(uplo, trans, n, k, alpha, a, offseta, lda, b, offsetb, ldb, beta, c, offsetc, ldc);
  }

  public void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int lda, double beta, double[] c, int ldc) {
    dsyrk(uplo, trans, n, k, alpha, a, 0, lda, beta, c, 0, ldc);
  }

  public void dsyrk(String uplo, String trans, int n, int k, double alpha, double[] a, int offseta, int lda, double beta, double[] c, int offsetc, int ldc) {
    f2j.dsyrk(uplo, trans, n, k, alpha, a, offseta, lda, beta, c, offsetc, ldc);
  }

  public void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int lda, float beta, float[] c, int ldc) {
    ssyrk(uplo, trans, n, k, alpha, a, 0, lda, beta, c, 0, ldc);
  }

  public void ssyrk(String uplo, String trans, int n, int k, float alpha, float[] a, int offseta, int lda, float beta, float[] c, int offsetc, int ldc) {
    f2j.ssyrk(uplo, trans, n, k, alpha, a, offseta, lda, beta, c, offsetc, ldc);
  }

  public void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx) {
    dtbmv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
  }

  public void dtbmv(String uplo, String trans, String diag, int n, int k, double[] a, int offseta, int lda, double[] x, int offsetx, int incx) {
    f2j.dtbmv(uplo, trans, diag, n, k, a, offseta, lda, x, offsetx, incx);
  }

  public void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx) {
    stbmv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
  }

  public void stbmv(String uplo, String trans, String diag, int n, int k, float[] a, int offseta, int lda, float[] x, int offsetx, int incx) {
    f2j.stbmv(uplo, trans, diag, n, k, a, offseta, lda, x, offsetx, incx);
  }

  public void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int lda, double[] x, int incx) {
    dtbsv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
  }

  public void dtbsv(String uplo, String trans, String diag, int n, int k, double[] a, int offseta, int lda, double[] x, int offsetx, int incx) {
    f2j.dtbsv(uplo, trans, diag, n, k, a, offseta, lda, x, offsetx, incx);
  }

  public void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int lda, float[] x, int incx) {
    stbsv(uplo, trans, diag, n, k, a, 0, lda, x, 0, incx);
  }

  public void stbsv(String uplo, String trans, String diag, int n, int k, float[] a, int offseta, int lda, float[] x, int offsetx, int incx) {
    f2j.stbsv(uplo, trans, diag, n, k, a, offseta, lda, x, offsetx, incx);
  }

  public void dtpmv(String uplo, String trans, String diag, int n, double[] a, double[] x, int incx) {
    dtpmv(uplo, trans, diag, n, a, 0, x, 0, incx);
  }

  public void dtpmv(String uplo, String trans, String diag, int n, double[] a, int offseta, double[] x, int offsetx, int incx) {
    f2j.dtpmv(uplo, trans, diag, n, a, offseta, x, offsetx, incx);
  }

  public void stpmv(String uplo, String trans, String diag, int n, float[] a, float[] x, int incx) {
    stpmv(uplo, trans, diag, n, a, 0, x, 0, incx);
  }

  public void stpmv(String uplo, String trans, String diag, int n, float[] a, int offseta, float[] x, int offsetx, int incx) {
    f2j.stpmv(uplo, trans, diag, n, a, offseta, x, offsetx, incx);
  }

  public void dtpsv(String uplo, String trans, String diag, int n, double[] a, double[] x, int incx) {
    dtpsv(uplo, trans, diag, n, a, 0, x, 0, incx);
  }

  public void dtpsv(String uplo, String trans, String diag, int n, double[] a, int offseta, double[] x, int offsetx, int incx) {
    f2j.dtpsv(uplo, trans, diag, n, a, offseta, x, offsetx, incx);
  }

  public void stpsv(String uplo, String trans, String diag, int n, float[] a, float[] x, int incx) {
    stpsv(uplo, trans, diag, n, a, 0, x, 0, incx);
  }

  public void stpsv(String uplo, String trans, String diag, int n, float[] a, int offseta, float[] x, int offsetx, int incx) {
    f2j.stpsv(uplo, trans, diag, n, a, offseta, x, offsetx, incx);
  }

  public void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a, int lda, double[] b, int ldb) {
    dtrmm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
  }

  public void dtrmm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb) {
    f2j.dtrmm(side, uplo, transa, diag, m, n, alpha, a, offseta, lda, b, offsetb, ldb);
  }

  public void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a, int lda, float[] b, int ldb) {
    strmm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
  }

  public void strmm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb) {
    f2j.strmm(side, uplo, transa, diag, m, n, alpha, a, offseta, lda, b, offsetb, ldb);
  }

  public void dtrmv(String uplo, String trans, String diag, int n, double[] a, int lda, double[] x, int incx) {
    dtrmv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
  }

  public void dtrmv(String uplo, String trans, String diag, int n, double[] a, int offseta, int lda, double[] x, int offsetx, int incx) {
    f2j.dtrmv(uplo, trans, diag, n, a, offseta, lda, x, offsetx, incx);
  }

  public void strmv(String uplo, String trans, String diag, int n, float[] a, int lda, float[] x, int incx) {
    strmv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
  }

  public void strmv(String uplo, String trans, String diag, int n, float[] a, int offseta, int lda, float[] x, int offsetx, int incx) {
    f2j.strmv(uplo, trans, diag, n, a, offseta, lda, x, offsetx, incx);
  }

  public void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a, int lda, double[] b, int ldb) {
    dtrsm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
  }

  public void dtrsm(String side, String uplo, String transa, String diag, int m, int n, double alpha, double[] a, int offseta, int lda, double[] b, int offsetb, int ldb) {
    f2j.dtrsm(side, uplo, transa, diag, m, n, alpha, a, offseta, lda, b, offsetb, ldb);
  }

  public void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a, int lda, float[] b, int ldb) {
    strsm(side, uplo, transa, diag, m, n, alpha, a, 0, lda, b, 0, ldb);
  }

  public void strsm(String side, String uplo, String transa, String diag, int m, int n, float alpha, float[] a, int offseta, int lda, float[] b, int offsetb, int ldb) {
    f2j.strsm(side, uplo, transa, diag, m, n, alpha, a, offseta, lda, b, offsetb, ldb);
  }

  public void dtrsv(String uplo, String trans, String diag, int n, double[] a, int lda, double[] x, int incx) {
    dtrsv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
  }

  public void dtrsv(String uplo, String trans, String diag, int n, double[] a, int offseta, int lda, double[] x, int offsetx, int incx) {
    f2j.dtrsv(uplo, trans, diag, n, a, offseta, lda, x, offsetx, incx);
  }

  public void strsv(String uplo, String trans, String diag, int n, float[] a, int lda, float[] x, int incx) {
    strsv(uplo, trans, diag, n, a, 0, lda, x, 0, incx);
  }

  public void strsv(String uplo, String trans, String diag, int n, float[] a, int offseta, int lda, float[] x, int offsetx, int incx) {
    f2j.strsv(uplo, trans, diag, n, a, offseta, lda, x, offsetx, incx);
  }

  public int idamax(int n, double[] x, int incx) {
    return idamax(n, x, 0, incx);
  }

  public int idamax(int n, double[] x, int offsetx, int incx) {
    if (n <= 0) {
      return -1;
    }
    if (incx <= 0) {
      return -1;
    }
    if (n == 1) {
      return 0;
    }
    int imax = 0;
    double max = x[offsetx];
    for (int i = 1, ix = incx; ix < n * incx; i += 1, ix += incx) {
      double val = Math.abs(x[offsetx + ix]);
      if (val > max) {
        imax = i;
        max = val;
      }
    }
    return imax + 1; // +1 because Fortran arrays are 1-indexed
  }

  public int isamax(int n, float[] x, int incx) {
    return isamax(n, x, 0, incx);
  }

  public int isamax(int n, float[] x, int offsetx, int incx) {
    if (n <= 0) {
      return -1;
    }
    if (incx <= 0) {
      return -1;
    }
    if (n == 1) {
      return 0;
    }
    int imax = 0;
    float max = x[offsetx];
    for (int i = 1, ix = incx; ix < n * incx; i += 1, ix += incx) {
      float val = Math.abs(x[offsetx + ix]);
      if (val > max) {
        imax = i;
        max = val;
      }
    }
    return imax + 1; // +1 because Fortran arrays are 1-indexed
  }

  public boolean lsame(String ca, String cb) {
    return ca != null && ca.length() == 1 && ca.equalsIgnoreCase(cb);
  }
}
