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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class DdotTest extends BLASTest {

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testSanity(BLAS blas) {
        assertEquals(f2j.ddot(M, dX, 1, dY, 1), blas.ddot(M, dX, 1, dY, 1), depsilon);
    }

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testOutOfBound(BLAS blas) {
        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> {
            blas.ddot(M + 1, dX, 1, dY, 1);
        });
    }

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testOutOfBoundBecauseOfOffset(BLAS blas) {
        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> {
            blas.ddot(M, dX, 1, 1, dY, 1, 1);
        });
    }

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testOutOfBoundOnlyForX(BLAS blas) {
        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> {
            blas.ddot(M, dX, 2, dY, 1);
        });
    }

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testXAndYAreNullAndNIsZero(BLAS blas) {
        assertEquals(0.0, blas.ddot(0, null, 1, null, 1));
    }

    @ParameterizedTest
    @MethodSource("BLASImplementations")
    void testXAndYAreNullAndNIsOne(BLAS blas) {
        assertThrows(java.lang.NullPointerException.class, () -> {
            blas.ddot(M, null, 1, null, 1);
        });
    }
}
