/*
 * Copyright 2020, 2021, Ludovic Henry
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
 *
 * Please contact git@ludovic.dev or visit ludovic.dev if you need additional
 * information or have any questions.
 */

package dev.ludovic.netlib.blas;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class SgemmTest extends BLASTest {

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testSanity(BLAS blas) {
        float[] expected, sgeCcopy;

        f2j.sgemm("N", "N", M, N, K, 1.0f, sgeA, M, sgeB, K, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N, K, 1.0f, sgeA, M, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N, K, 1.0f, sgeA, M, sgeBT, N, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N, K, 1.0f, sgeA, M, sgeBT, N, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N, K, 1.0f, sgeAT, K, sgeBT, N, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N, K, 1.0f, sgeAT, K, sgeBT, N, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M, N, K, 1.0f, sgeA, M, sgeB, K, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N, K, 1.0f, sgeA, M, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N, K, 1.0f, sgeA, M, sgeBT, N, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N, K, 1.0f, sgeA, M, sgeBT, N, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N, K, 1.0f, sgeAT, K, sgeBT, N, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N, K, 1.0f, sgeAT, K, sgeBT, N, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M, N, K, 0.0f, sgeA, M, sgeB, K, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N, K, 0.0f, sgeA, M, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N, K, 0.0f, sgeA, M, sgeBT, N, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N, K, 0.0f, sgeA, M, sgeBT, N, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N, K, 0.0f, sgeAT, K, sgeBT, N, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N, K, 0.0f, sgeAT, K, sgeBT, N, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M/2, N, K, 1.0f, sgeA, M/2, sgeB, K, 2.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "N", M/2, N, K, 1.0f, sgeA, M/2, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M/2, N, K, 1.0f, sgeA, M/2, sgeBT, N, 2.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "T", M/2, N, K, 1.0f, sgeA, M/2, sgeBT, N, 2.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M/2, N, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "N", M/2, N, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M/2, N, K, 1.0f, sgeAT, K, sgeBT, N, 2.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "T", M/2, N, K, 1.0f, sgeAT, K, sgeBT, N, 2.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M/2, N, K, 1.0f, sgeA, M/2, sgeB, K, 0.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "N", M/2, N, K, 1.0f, sgeA, M/2, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M/2, N, K, 1.0f, sgeA, M/2, sgeBT, N, 0.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "T", M/2, N, K, 1.0f, sgeA, M/2, sgeBT, N, 0.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M/2, N, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "N", M/2, N, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M/2, N, K, 1.0f, sgeAT, K, sgeBT, N, 0.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "T", M/2, N, K, 1.0f, sgeAT, K, sgeBT, N, 0.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M/2, N, K, 0.0f, sgeA, M/2, sgeB, K, 1.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "N", M/2, N, K, 0.0f, sgeA, M/2, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M/2, N, K, 0.0f, sgeA, M/2, sgeBT, N, 1.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("N", "T", M/2, N, K, 0.0f, sgeA, M/2, sgeBT, N, 1.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M/2, N, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "N", M/2, N, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M/2, N, K, 0.0f, sgeAT, K, sgeBT, N, 1.0f, expected = sgeC.clone(), M/2);
        blas.sgemm("T", "T", M/2, N, K, 0.0f, sgeAT, K, sgeBT, N, 1.0f, sgeCcopy = sgeC.clone(), M/2);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M, N/2, K, 1.0f, sgeA, M, sgeB, K, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N/2, K, 1.0f, sgeA, M, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N/2, K, 1.0f, sgeA, M, sgeBT, N/2, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N/2, K, 1.0f, sgeA, M, sgeBT, N/2, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N/2, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N/2, K, 1.0f, sgeAT, K, sgeB, K, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N/2, K, 1.0f, sgeAT, K, sgeBT, N/2, 2.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N/2, K, 1.0f, sgeAT, K, sgeBT, N/2, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M, N/2, K, 1.0f, sgeA, M, sgeB, K, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N/2, K, 1.0f, sgeA, M, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N/2, K, 1.0f, sgeA, M, sgeBT, N/2, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N/2, K, 1.0f, sgeA, M, sgeBT, N/2, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N/2, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N/2, K, 1.0f, sgeAT, K, sgeB, K, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N/2, K, 1.0f, sgeAT, K, sgeBT, N/2, 0.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N/2, K, 1.0f, sgeAT, K, sgeBT, N/2, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "N", M, N/2, K, 0.0f, sgeA, M, sgeB, K, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "N", M, N/2, K, 0.0f, sgeA, M, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("N", "T", M, N/2, K, 0.0f, sgeA, M, sgeBT, N/2, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("N", "T", M, N/2, K, 0.0f, sgeA, M, sgeBT, N/2, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "N", M, N/2, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "N", M, N/2, K, 0.0f, sgeAT, K, sgeB, K, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.sgemm("T", "T", M, N/2, K, 0.0f, sgeAT, K, sgeBT, N/2, 1.0f, expected = sgeC.clone(), M);
        blas.sgemm("T", "T", M, N/2, K, 0.0f, sgeAT, K, sgeBT, N/2, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);
    }
}
