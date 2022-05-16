module dev.ludovic.netlib.blas {
    exports dev.ludovic.netlib.blas;

    requires java.logging;

    requires jdk.incubator.vector;

    requires transitive arpack.combined.all;
}