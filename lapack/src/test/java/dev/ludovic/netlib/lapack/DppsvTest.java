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
 */

import dev.ludovic.netlib.LAPACK;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

public class DppsvTest extends LAPACKTest {

    @ParameterizedTest
    @MethodSource("LAPACKImplementations")
    void testSanity(LAPACK lapack) {
        double[] dgeAUexpected, dgeAUobtained, dgeBexpected, dgeBobtained;
        org.netlib.util.intW infoexpected, infoobtained;

        f2j.dppsv("U", N, M, dgeAUexpected = dgeAU.clone(), dgeBexpected = dgeB.clone(), N, infoexpected = new org.netlib.util.intW(0));
        lapack.dppsv("U", N, M, dgeAUobtained = dgeAU.clone(), dgeBobtained = dgeB.clone(), N, infoobtained = new org.netlib.util.intW(0));
        assertEquals(2, infoexpected.val);
        assertEquals(2, infoobtained.val);
        assertArrayEquals(dgeAUexpected, dgeAUobtained, depsilon);
        assertArrayEquals(dgeBexpected, dgeBobtained, depsilon);
    }
}