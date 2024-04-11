
import _qedhf
import numpy

class qed(object):


    def __init__(self):
        self.nao = 10
        self.nmodes = 2
        self.gmat = numpy.ones((self.nmodes, self.nao, self.nao))
        self.eta = numpy.ones(self.nao)
        self.g = numpy.zeros((self.nao, self.nao))

        for i in range(self.nao):
            self.g[i,i] = i * 1.0

    def cpp_method(self):
        g2 = _qedhf.cpp_method_impl(self, self.g)

        #self.g = _my_module.cpp_method_impl(self.g)
        print(g2)

    def gaussian_factor(self, w, p, q, r, s):
        return _qedhf.gaussian_factor_vt_qedhf(w, p, q, r, s)

    def test_gaussian(self, N, freq, eta):
        return _qedhf.test_gaussian(N, freq, eta)

    def test_eigen3(self):
        mat = _qedhf.eigen3_exampleFunction()
        print(f"eigen3 mat is", mat)


    def get_gaussian_factor(self, freq, eta_p, eta_q, eta_r=None, eta_s=None):
        half = 0.5

        # Check if r and s are provided or default
        if eta_r and eta_s:
            return numpy.exp(-half * ((eta_p + eta_r - eta_s - eta_q) / freq) ** 2)
        else:
            return numpy.exp(-half * ((eta_p - eta_q) / freq) ** 2)


if __name__ == "__main__":

    import time
    obj = qed()
    obj.cpp_method()

    obj.test_eigen3()

    freq = 1.0
    eta_p = 1.0
    eta_q = 2.0
    eta_r = 3.0
    eta_s = 4.0

    for N in range(15, 15, 5):

        gaussian1 = numpy.zeros(N**4)
        gaussian2 = numpy.zeros(N**4)
        eta = numpy.random.rand(N)

        nrun = 5
        t1 = 0.0
        t2 = 0.0
        for irun in range(nrun):
            t0 = time.time()
            count = 0
            for p in range(N):
                for q in range(N):
                    for r in range(N):
                        for s in range(N):
                            gaussian1[count] = obj.gaussian_factor(freq, eta[p], eta[q], eta[r], eta[s])
                            count += 1
            t1 += time.time() - t0

            t0 = time.time()
            gaussian2 = obj.test_gaussian(N, freq, eta)
            t2 += time.time() - t0
            print("Guaissian factors are the same?", numpy.allclose(gaussian1, gaussian2))
        # end of nrun

        t1 /= nrun
        t2 /= nrun
        print(f"Size {N:3d} Wall times: {t1:.3f} {t2:.3f} cpp speadup= {t1/t2:.2f}")

