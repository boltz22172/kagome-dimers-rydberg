import math
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

N=50

class MPS:
    def __init__(self, tensors, dtype=torch.complex64):
        """
        tensors: list of torch tensors, each of shape (d, D, D)
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors.clone() for _ in range(5)]  # 默认N=5个site
        self.tensors = [t.to(dtype) for t in tensors]
        self.L = len(tensors)
        self.D = tensors[0].shape[1]
        self.dtype = dtype

    def boundary_vec(self):
        """Return all-ones boundary vector |b>"""
        return torch.ones(self.D, dtype=self.dtype)

    def norm(self):
        """Compute <ψ|ψ> with boundary ones"""
        vL = self.boundary_vec()
        vR = self.boundary_vec()
        E = torch.outer(vL.conj(), vL)

        for A in self.tensors:
            E = torch.einsum('ab, sai, sbj -> ij', E, A.conj(), A)

        norm = torch.einsum('a,ab,b->', vR.conj(), E, vR)
        return norm

    def expectation(self, ops):
        """
        Compute <ψ| (⊗_i ops[i]) |ψ>, where each ops[i] is (d,d)
        If ops is a single matrix, use it on all sites.
        """
        vL = self.boundary_vec()
        vR = self.boundary_vec()
        E = torch.outer(vL.conj(), vL)

        if isinstance(ops, torch.Tensor):
            ops = [ops for _ in range(self.L)]

        for A, O in zip(self.tensors, ops):
            E = torch.einsum('ab, sai, pbj, ps -> ij', E, A.conj(), A, O)

        val = torch.einsum('a,ab,b->', vR.conj(), E, vR)
        return val

    def overlap(self, other):
        """
        Compute <self|other>, assuming both MPS have the same bond structure.
        """
        assert self.L == other.L, "MPS lengths must match for overlap"
        assert self.D == other.D, "Bond dimensions must match"

        vL = self.boundary_vec()
        vR = self.boundary_vec()
        E = torch.outer(vL.conj(), vL)

        for A1, A2 in zip(self.tensors, other.tensors):
            E = torch.einsum('ab, sai, sbj -> ij', E, A1.conj(), A2)

        val = torch.einsum('a,ab,b->', vR.conj(), E, vR)
        return val

def gauss_pos_random(sigma):
    while True:
        x = torch.normal(0.0, sigma, size=(1,))
        if x.item() > 0:
            return x.item()

def error(sigma):
    return gauss_pos_random(sigma)

def sample_function(L_total, sigma, i_start, L_c):

    z_string = torch.diag(torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1], dtype=torch.complex64))
    d, D = 11, 4
    A_list = []
    for n in range(N):
        A = torch.zeros(d, D, D, dtype=torch.complex64)
        A[0] = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, error(sigma), 0]], dtype=torch.complex64)
        A[1] = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, error(sigma)]], dtype=torch.complex64)
        A[2] = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, error(sigma), 0]], dtype=torch.complex64)
        A[3] = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, error(sigma)]], dtype=torch.complex64)
        A[4] = torch.tensor([[error(sigma), 0, 0, 0], [error(sigma), 0, 0, 0], [1, 0, 0, 0], [error(sigma), 0, 0, 0]], dtype=torch.complex64)
        A[5] = torch.tensor([[0, error(sigma), 0, 0], [0, error(sigma), 0, 0], [0, 1, 0, 0], [0, error(sigma), 0, 0]], dtype=torch.complex64)
        A[6] = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.complex64)
        A[7] = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.complex64)
        A[8] = torch.tensor([[0, error(sigma), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, error(sigma), 0, 0]], dtype=torch.complex64)
        A[9] = torch.tensor([[0, 0, 0, 0], [error(sigma), 0, 0, 0], [0, 0, 0, 0], [error(sigma), 0, 0, 0]], dtype=torch.complex64)
        A[10] = torch.tensor([[0, 0, 0, error(sigma)], [0, 0, 0, error(sigma)], [0, 0, 0, error(sigma)], [0, 0, 0, error(sigma)]], dtype=torch.complex64)
        A_list.append(A)

    psi = MPS(A_list)
    norm_MPS = psi.norm().item()
    op_list = []
    for i in range(N):
        if i == i_start or i == i_start + L_c:
            op_list.append(z_string)
        else:
            op_list.append(torch.eye(11, dtype=torch.complex64))
    correlation = psi.expectation(op_list).item()
    correlation_normalized = correlation / norm_MPS
    return correlation_normalized

def monte_carlo(L_total, sigma, i_start, L_c, n_samples=1000):
    results = []
    for _ in range(n_samples):
        result = sample_function(L_total, sigma, i_start, L_c)
        results.append(result.real)
    average = sum(results) / n_samples
    variance = sum((x - average)**2 for x in results) / (n_samples - 1)
    std_error = (variance**0.5) / (n_samples**0.5)
    return results, average, std_error

def fidelity_sample(L_total, sigma):
    d, D = 11, 4
    N = L_total

    def construct_A_list(sigma):
        A_list = []
        for n in range(N):
            A = torch.zeros(d, D, D, dtype=torch.complex64)
            A[0] = torch.tensor([[0, 0, 1, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, error(sigma), 0]], dtype=torch.complex64)
            A[1] = torch.tensor([[0, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, error(sigma)]], dtype=torch.complex64)
            A[2] = torch.tensor([[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, error(sigma), 0]], dtype=torch.complex64)
            A[3] = torch.tensor([[0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, error(sigma)]], dtype=torch.complex64)
            A[4] = torch.tensor([[error(sigma), 0, 0, 0],
                                 [error(sigma), 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [error(sigma), 0, 0, 0]], dtype=torch.complex64)
            A[5] = torch.tensor([[0, error(sigma), 0, 0],
                                 [0, error(sigma), 0, 0],
                                 [0, 1, 0, 0],
                                 [0, error(sigma), 0, 0]], dtype=torch.complex64)
            A[6] = torch.tensor([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 0]], dtype=torch.complex64)
            A[7] = torch.tensor([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=torch.complex64)
            A[8] = torch.tensor([[0, error(sigma), 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, error(sigma), 0, 0]], dtype=torch.complex64)
            A[9] = torch.tensor([[0, 0, 0, 0],
                                 [error(sigma), 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [error(sigma), 0, 0, 0]], dtype=torch.complex64)
            A[10] = torch.tensor([[0, 0, 0, error(sigma)],
                                  [0, 0, 0, error(sigma)],
                                  [0, 0, 0, error(sigma)],
                                  [0, 0, 0, error(sigma)]], dtype=torch.complex64)
            A_list.append(A)
        return A_list

    psi_sigma = MPS(construct_A_list(sigma))

    A_list_ideal = []
    for n in range(N):
        A = torch.zeros(d, D, D, dtype=torch.complex64)
        A[0] = torch.tensor([[0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[1] = torch.tensor([[0, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[2] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[3] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[4] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[5] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[6] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 0]], dtype=torch.complex64)
        A[7] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0]], dtype=torch.complex64)
        A[8] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[9] = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=torch.complex64)
        A[10] = torch.tensor([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]], dtype=torch.complex64)
        A_list_ideal.append(A)

    psi_ideal = MPS(A_list_ideal)

    overlap = psi_sigma.overlap(psi_ideal)
    norm_ideal = psi_ideal.norm()
    norm_sigma = psi_sigma.norm()
    overlap /= torch.sqrt(norm_ideal * norm_sigma)
    fidelity = torch.abs(overlap)**2
    return fidelity.item()

def monte_carlo_fidelity(L_total, sigma, n_samples=1000):
    results = []
    for _ in range(n_samples):
        fidelity = fidelity_sample(L_total, sigma)
        results.append(fidelity)
    average = sum(results) / n_samples
    variance = sum((x - average)**2 for x in results) / (n_samples - 1)
    std_error = (variance**0.5) / (n_samples**0.5)
    return results, average, std_error

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    sigmas = [0.001, 0.01, 0.05, 0.1, 0.2]
    L_c_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    L_total = 30
    i_start = 5
    n_samples = 100

    avg_mat = np.zeros((len(sigmas), len(L_c_list)), dtype=float)
    err_mat = np.zeros_like(avg_mat)
    rows1 = []

    plt.figure(figsize=(8, 5))
    for si, sigma in enumerate(sigmas):
        avgs = []
        errors = []
        for lj, L_c in enumerate(L_c_list):
            _, avg, std_error = monte_carlo(L_total, sigma, i_start, L_c, n_samples=n_samples)
            avgs.append(avg)
            errors.append(std_error)
            avg_mat[si, lj] = avg
            err_mat[si, lj] = std_error
            rows1.append([sigma, L_c, avg, std_error])

        plt.errorbar(
            L_c_list, avgs, yerr=errors,
            fmt='-o', markersize=4, capsize=3, elinewidth=1,
            label=fr'$\sigma={sigma}$'
        )

    write_csv("corr_data.csv", ["sigma", "L_c", "avg", "std_error"], rows1)

    plt.xlabel(r"$L_c$", fontsize=16)
    plt.ylabel(r"$\langle \mathcal{Z}_i \mathcal{Z}_{i+L_c}\rangle$", fontsize=16)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    sigmas = np.linspace(0.01, 0.1, 20)
    L_totals = [15, 20, 25, 30]
    n_samples = 50

    avg_mat = np.zeros((len(L_totals), len(sigmas)), dtype=float)
    err_mat = np.zeros_like(avg_mat)
    rows2 = []

    plt.figure(figsize=(8, 5))
    for iL, L_total in enumerate(L_totals):
        print(f"Processing L_total={L_total}...")
        avgs = []
        errors = []
        for isg, sigma in enumerate(sigmas):
            _, avg, std_error = monte_carlo_fidelity(L_total, float(sigma), n_samples=n_samples)
            avgs.append(avg)
            errors.append(std_error)
            avg_mat[iL, isg] = avg
            err_mat[iL, isg] = std_error
            rows2.append([L_total, float(sigma), avg, std_error])

        plt.errorbar(
            sigmas, avgs, yerr=errors,
            fmt='-o', markersize=4, capsize=3, elinewidth=1,
            label=fr'$L_{{\mathrm{{total}}}}={L_total}$'
        )

    write_csv("fidelity_data.csv", ["L_total", "sigma", "avg", "std_error"], rows2)

    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel(r"$\langle |\langle \psi_\sigma | \psi_0 \rangle|^2 \rangle$", fontsize=16)
    plt.grid(True, alpha=0.4)
    plt.legend(title=r"$L_{\mathrm{total}}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()