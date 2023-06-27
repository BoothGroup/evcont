from pyscf.mcscf.casci import CASCI


class CustomCASCI(CASCI):
    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        assert not self.natorb
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        self.check_sanity()

        ncas = self.ncas
        nelecas = self.nelecas

        # 2e
        eri_cas = self.get_h2eff(self.mo_coeff)

        # 1e
        h1eff, energy_core = self.get_h1eff(self.mo_coeff)

        if h1eff.shape[0] != ncas:
            raise RuntimeError(
                "Active space size error. nmo=%d ncore=%d ncas=%d"
                % (mo_coeff.shape[1], self.ncore, ncas)
            )

        self.fcisolver.mo_coeff = self.mo_coeff
        self.fcisolver.mol = self.mol
        self.e_tot, self.ci = self.fcisolver.kernel(
            h1eff,
            eri_cas,
            ncas,
            nelecas,
            ecore=energy_core,
        )

        self.e_cas = self.e_tot - energy_core

        if self.canonicalization:
            self.canonicalize_(
                mo_coeff,
                self.ci,
                sort=self.sorting_mo_energy,
                cas_natorb=self.natorb,
            )
        self.fcisolver.mo_coeff = self.mo_coeff

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
