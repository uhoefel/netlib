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

class SsymmTest extends BLASTest {

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testSanity(BLAS blas) {
        float[] expected, sgeCcopy;

        f2j.ssymm("L", "U", M, N, 1.0f, ssyA, M, sgeB, M, 2.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "U", M, N, 1.0f, ssyA, M, sgeB, M, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("L", "L", M, N, 1.0f, ssyA, M, sgeB, M, 2.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "L", M, N, 1.0f, ssyA, M, sgeB, M, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "U", M, N, 1.0f, ssyA, N, sgeB, M, 2.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "U", M, N, 1.0f, ssyA, N, sgeB, M, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "L", M, N, 1.0f, ssyA, N, sgeB, M, 2.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "L", M, N, 1.0f, ssyA, N, sgeB, M, 2.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("L", "U", M, N, 1.0f, ssyA, M, sgeB, M, 0.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "U", M, N, 1.0f, ssyA, M, sgeB, M, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("L", "L", M, N, 1.0f, ssyA, M, sgeB, M, 0.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "L", M, N, 1.0f, ssyA, M, sgeB, M, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "U", M, N, 1.0f, ssyA, N, sgeB, M, 0.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "U", M, N, 1.0f, ssyA, N, sgeB, M, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "L", M, N, 1.0f, ssyA, N, sgeB, M, 0.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "L", M, N, 1.0f, ssyA, N, sgeB, M, 0.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("L", "U", M, N, 0.0f, ssyA, M, sgeB, M, 1.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "U", M, N, 0.0f, ssyA, M, sgeB, M, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("L", "L", M, N, 0.0f, ssyA, M, sgeB, M, 1.0f, expected = sgeC.clone(), M);
        blas.ssymm("L", "L", M, N, 0.0f, ssyA, M, sgeB, M, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "U", M, N, 0.0f, ssyA, N, sgeB, M, 1.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "U", M, N, 0.0f, ssyA, N, sgeB, M, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);

        f2j.ssymm("R", "L", M, N, 0.0f, ssyA, N, sgeB, M, 1.0f, expected = sgeC.clone(), M);
        blas.ssymm("R", "L", M, N, 0.0f, ssyA, N, sgeB, M, 1.0f, sgeCcopy = sgeC.clone(), M);
        assertArrayEquals(expected, sgeCcopy, sepsilon);
    }
}
