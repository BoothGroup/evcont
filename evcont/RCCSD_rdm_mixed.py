from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def make_rdm1_f(t1a=None, t1b=None, t2a=None, t2b=None, l1a=None, l2a=None, **kwargs):
    nocc, nvir = t1a.shape
    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 1RDM
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->jiab", t2b) * 2
    x0 += einsum("ijab->jiba", t2b) * -1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=np.float64)
    rdm1_f_oo += einsum("abij,ikba->jk", l2a, x0) * -2
    del x0
    x1 = np.zeros((nocc, nvir), dtype=np.float64)
    x1 += einsum("ia->ia", t1a)
    x1 -= einsum("ia->ia", t1b)
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 -= einsum("abij->jiab", l2a)
    x2 += einsum("abij->jiba", l2a) * 2
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum("ia,ijab->jb", x1, x2)
    del x1
    del x2
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 -= einsum("ia->ia", x3)
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=np.float64)
    rdm1_f_ov -= einsum("ia->ia", x3) * 2
    del x3
    x4 += einsum("ai->ia", l1a)
    rdm1_f_oo -= einsum("ia,ja->ji", t1b, x4) * 2
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=np.float64)
    rdm1_f_vv += einsum("ia,ib->ab", t1b, x4) * 2
    x5 = 0
    x5 += einsum("abij,ijba->", l2a, t2a)
    x14 = 0
    x14 += 0.25*x5
    x20 = 0
    x20 += -0.5*x5
    del x5
    x6 = 0
    x6 += einsum("abij,ijab->", l2a, t2b)
    x14 += 0.5*x6
    x20 += -1.0*x6
    del x6
    x7 = 0
    x7 += einsum("abij,ijab->", l2a, t2a)
    x14 += -0.5*x7
    x20 += 1.0*x7
    del x7
    x8 = 0
    x8 += einsum("abij,ijba->", l2a, t2b)
    x14 += -0.25*x8
    x20 += 0.5*x8
    del x8
    x9 = np.zeros((nocc, nvir), dtype=np.float64)
    x9 += einsum("ia->ia", t1a)
    x9 += einsum("ia->ia", t1b) * -1
    x14 += einsum("ai,ia->", l1a, x9) * -0.5
    x20 += einsum("ai,ia->", l1a, x9)
    del x9
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=np.float64)
    rdm1_f_vo += einsum("ia->ai", t1b) * -4.0*x20
    del x20
    x10 = np.zeros((nocc, nvir), dtype=np.float64)
    x10 += einsum("ia->ia", t1b)
    x10 += einsum("ia->ia", t1a) * -0.5
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum("abij->jiab", l2a) * -0.5
    x11 += einsum("abij->jiba", l2a)
    x12 = np.zeros((nocc, nvir), dtype=np.float64)
    x12 += einsum("ia,ijab->jb", t1a, x11)
    x14 += einsum("ia,ia->", x10, x12) * -1
    del x10
    del x12
    x13 = np.zeros((nocc, nvir), dtype=np.float64)
    x13 += einsum("ia,ijab->jb", t1b, x11)
    x14 += einsum("ia,ia->", t1b, x13) * 0.5
    del x13
    rdm1_f_oo += einsum("ij->ji", delta_oo) * 8.0*x14
    del x14
    rdm1_f_vv += einsum("ijab,ijac->bc", t2b, x11) * 4
    del x11
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x15 += einsum("ia,abjk->kjib", t1b, l2a)
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x16 += einsum("ijka->ijka", x15) * -1
    x16 += einsum("ijka->jika", x15) * 2
    del x15
    rdm1_f_vo += einsum("ijab,jika->bk", t2b, x16) * -2
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 -= einsum("ijab->jiab", t2b)
    x17 += einsum("ijab->jiba", t2b) * 2
    rdm1_f_vo += einsum("ia,ijab->bj", x4, x17) * 2
    del x17
    del x4
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum("ijab->jiab", t2b)
    x18 += einsum("ijab->jiba", t2b) * -0.5
    x19 = np.zeros((nocc, nocc), dtype=np.float64)
    x19 += einsum("abij,ikba->jk", l2a, x18) * 2
    del x18
    x19 += einsum("ai,ja->ij", l1a, t1b)
    rdm1_f_vo += einsum("ia,ij->aj", t1b, x19) * -2
    del x19
    rdm1_f_oo += einsum("ij->ji", delta_oo) * 2
    rdm1_f_ov += einsum("ai->ia", l1a) * 2
    rdm1_f_vo += einsum("ia->ai", t1b) * 2

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(t1a=None, t1b=None, t2a=None, t2b=None, l1a=None, l2a=None, **kwargs):
    nocc, nvir = t1a.shape
    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 2RDM
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x0 += einsum("abij,klab->ijkl", l2a, t2b)
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x34 += einsum("ia,jikl->jkla", t1b, x0)
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_ooov += einsum("ijka->jika", x34) * 4
    rdm2_f_ooov += einsum("ijka->kija", x34) * -2
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_ovoo += einsum("ijka->jaki", x34) * -2
    rdm2_f_ovoo += einsum("ijka->kaji", x34) * 4
    del x34
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_oooo += einsum("ijkl->kjli", x0) * -2
    rdm2_f_oooo += einsum("ijkl->ljki", x0) * 4
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum("ia,abjk->kjib", t1b, l2a)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x2 += einsum("ia,jkla->jkil", t1b, x1)
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x29 += einsum("ia,ijkl->jlka", t1b, x2)
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x32 += einsum("ijka->ijka", x29)
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x60 += einsum("ia,ijkb->jkab", t1b, x29)
    del x29
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x64 += einsum("ijab->ijab", x60)
    del x60
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x67 += einsum("ijab,jikl->lkab", t2b, x2)
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x78 += einsum("ijab->ijab", x67) * -4.000000000000003
    del x67
    rdm2_f_oooo -= einsum("ijkl->kjli", x2) * 2
    rdm2_f_oooo += einsum("ijkl->ljki", x2) * 4
    del x2
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x24 += einsum("ijab,kjla->klib", t2b, x1)
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x27 -= einsum("ijka->ijka", x24)
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x53 -= einsum("ijka->ijka", x24)
    del x24
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x28 += einsum("ijab,jkla->klib", t2b, x1)
    x32 += einsum("ijka->ijka", x28)
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x48 += einsum("ijka->ijka", x28)
    del x28
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x30 -= einsum("ijka->ijka", x1)
    x30 += einsum("ijka->jika", x1) * 2
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x31 += einsum("ijab,kila->kljb", t2b, x30)
    x32 -= einsum("ijka->ijka", x31)
    rdm2_f_ooov += einsum("ijka->jika", x32) * 4
    rdm2_f_ooov -= einsum("ijka->kija", x32) * 2
    rdm2_f_ovoo -= einsum("ijka->jaki", x32) * 2
    rdm2_f_ovoo += einsum("ijka->kaji", x32) * 4
    del x32
    x48 -= einsum("ijka->ijka", x31)
    del x31
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x49 += einsum("ia,ijkb->jkab", t1b, x48)
    del x48
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x52 += einsum("ijab->ijab", x49)
    del x49
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x92 += einsum("ia,jikb->jkab", t1b, x30)
    del x30
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_oovv -= einsum("ijab->jiba", x92) * 2
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_vvoo -= einsum("ijab->baji", x92) * 2
    del x92
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x40 += einsum("ijka->ijka", x1) * -1
    x40 += einsum("ijka->jika", x1) * 2
    x41 = np.zeros((nocc, nvir), dtype=np.float64)
    x41 += einsum("ijab,jika->kb", t2b, x40) * 0.5
    del x40
    x45 = np.zeros((nocc, nvir), dtype=np.float64)
    x45 += einsum("ia->ia", x41)
    x83 = np.zeros((nocc, nvir), dtype=np.float64)
    x83 += einsum("ia->ia", x41)
    del x41
    x98 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x98 += einsum("ijka->ijka", x1) * 2
    x98 -= einsum("ijka->jika", x1)
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x99 += einsum("ia,jikb->jkab", t1b, x98)
    del x98
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_ovvo -= einsum("ijab->jabi", x99) * 2
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_voov -= einsum("ijab->bija", x99) * 2
    del x99
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x108 += einsum("ia,jikb->jkba", t1b, x1)
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x109 += einsum("ijab->ijab", x108)
    del x108
    x111 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x111 += einsum("ijab,jikc->kcba", t2b, x1)
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_ovvv += einsum("iabc->ibac", x111) * 2
    rdm2_f_ovvv += einsum("iabc->icab", x111) * -4
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_vvov += einsum("iabc->abic", x111) * -4
    rdm2_f_vvov += einsum("iabc->acib", x111) * 2
    del x111
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_oovo += einsum("ijka->kiaj", x1) * 2
    rdm2_f_oovo -= einsum("ijka->kjai", x1) * 4
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_vooo -= einsum("ijka->aikj", x1) * 4
    rdm2_f_vooo += einsum("ijka->ajki", x1) * 2
    del x1
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum("ia,baji->jb", t1a, l2a)
    x4 = 0
    x4 += einsum("ia,ia->", t1a, x3)
    del x3
    x10 = 0
    x10 += 2.0*x4
    x44 = 0
    x44 += 1.0*x4
    x86 = 0
    x86 += 1.0*x4
    del x4
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 += einsum("ia,abij->jb", t1b, l2a)
    x6 = 0
    x6 += einsum("ia,ia->", t1b, x5)
    del x5
    x10 += 2.0*x6
    x44 += 1.0*x6
    x86 += 1.0*x6
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum("ijab->jiba", t2a) * 2
    x7 += einsum("ijab->jiba", t2b) * -2
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum("ijab->ijba", x7)
    x8 += einsum("ijab->ijab", x7) * -0.5
    del x7
    x8 += einsum("ia,jb->ijab", t1a, t1a)
    x8 += einsum("ia,jb->ijab", t1b, t1b)
    x9 = 0
    x9 += einsum("abij,ijba->", l2a, x8)
    x10 += -1.0*x9
    del x9
    rdm2_f_oooo += einsum("ij,kl->lijk", delta_oo, delta_oo) * -2.0*x10
    rdm2_f_oooo += einsum("ij,kl->jilk", delta_oo, delta_oo) * 4.0*x10
    del x10
    x43 = 0
    x43 += einsum("abij,ijba->", l2a, x8) * 0.5
    del x8
    x44 += -1.0*x43
    del x43
    x45 += einsum("ia->ia", t1b) * -1.0*x44
    del x44
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 -= einsum("abij->jiab", l2a)
    x11 += einsum("abij->jiba", l2a) * 2
    x12 = np.zeros((nocc, nvir), dtype=np.float64)
    x12 += einsum("ia,ijab->jb", t1b, x11)
    x13 = 0
    x13 += einsum("ia,ia->", t1a, x12)
    del x12
    x39 = np.zeros((nocc, nvir), dtype=np.float64)
    x39 += einsum("ia->ia", t1b) * 2*x13
    rdm2_f_oooo += einsum("ij,kl->lijk", delta_oo, delta_oo) * 4*x13
    rdm2_f_oooo -= einsum("ij,kl->jilk", delta_oo, delta_oo) * 8*x13
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x46 += einsum("ijab,ikbc->kjca", t2b, x11)
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x47 += einsum("ijab,ikac->kjcb", t2b, x46)
    del x46
    x52 -= einsum("ijab->jiba", x47)
    del x47
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x61 += einsum("ijab,ikac->kjcb", t2b, x11)
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x62 += einsum("ijab,ikac->kjcb", t2b, x61)
    x64 += einsum("ijab->jiba", x62) * 2
    del x62
    x109 -= einsum("ijab->ijab", x61)
    rdm2_f_vvoo -= einsum("ijab->abji", x61) * 2
    del x61
    x14 = np.zeros((nocc, nocc), dtype=np.float64)
    x14 += einsum("ai,ja->ij", l1a, t1b)
    x23 = np.zeros((nocc, nocc), dtype=np.float64)
    x23 += einsum("ij->ij", x14)
    x57 = np.zeros((nocc, nvir), dtype=np.float64)
    x57 += einsum("ia,ij->ja", t1b, x14)
    x89 = np.zeros((nocc, nvir), dtype=np.float64)
    x89 -= einsum("ia->ia", x57)
    x90 = np.zeros((nocc, nvir), dtype=np.float64)
    x90 += einsum("ia->ia", x57)
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_ovov -= einsum("ia,jb->iajb", t1b, x57) * 4
    rdm2_f_ovov += einsum("ia,jb->jaib", t1b, x57) * 2
    del x57
    rdm2_f_oooo -= einsum("ij,kl->jilk", delta_oo, x14) * 4
    rdm2_f_oooo += einsum("ij,kl->lijk", delta_oo, x14) * 2
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum("ijab->jiab", t2b)
    x15 += einsum("ijab->jiba", t2b) * -0.5
    x16 = np.zeros((nocc, nocc), dtype=np.float64)
    x16 += einsum("abij,ikba->jk", l2a, x15)
    rdm2_f_oooo += einsum("ij,kl->jilk", delta_oo, x16) * -8
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x16) * 4
    rdm2_f_oooo += einsum("ij,kl->ljik", delta_oo, x16) * 4
    rdm2_f_oooo += einsum("ij,kl->lkij", delta_oo, x16) * -8
    del x16
    x33 = np.zeros((nocc, nocc), dtype=np.float64)
    x33 += einsum("abij,ikba->jk", l2a, x15) * 2
    del x15
    x42 = np.zeros((nocc, nvir), dtype=np.float64)
    x42 += einsum("ia,ij->ja", t1b, x33) * 0.5
    x45 += einsum("ia->ia", x42)
    rdm2_f_ooov += einsum("ij,ka->jika", delta_oo, x45) * -8
    rdm2_f_ooov += einsum("ij,ka->kjia", delta_oo, x45) * 4
    rdm2_f_ovoo += einsum("ij,ka->jaki", delta_oo, x45) * 4
    rdm2_f_ovoo += einsum("ij,ka->kaij", delta_oo, x45) * -8
    del x45
    x83 += einsum("ia->ia", x42)
    del x42
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x84 += einsum("ia,jb->ijab", t1b, x83) * 8.000000000000005
    del x83
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x82 += einsum("ij,ikab->jkab", x33, t2b) * 4
    x84 += einsum("ijab->jiba", x82)
    del x82
    rdm2_f_ooov += einsum("ia,jk->ijka", t1b, x33) * 2
    rdm2_f_ooov += einsum("ia,jk->kjia", t1b, x33) * -4
    rdm2_f_ovoo += einsum("ia,jk->iakj", t1b, x33) * -4
    rdm2_f_ovoo += einsum("ia,jk->kaij", t1b, x33) * 2
    del x33
    x17 = np.zeros((nocc, nvir), dtype=np.float64)
    x17 += einsum("ia->ia", t1a)
    x17 -= einsum("ia->ia", t1b)
    x18 = np.zeros((nocc, nvir), dtype=np.float64)
    x18 += einsum("ia,ijab->jb", x17, x11)
    del x17
    x19 = np.zeros((nocc, nocc), dtype=np.float64)
    x19 += einsum("ia,ja->ij", t1b, x18)
    x27 -= einsum("ia,jk->kija", t1b, x19)
    x50 = np.zeros((nocc, nvir), dtype=np.float64)
    x50 -= einsum("ia,ji->ja", t1b, x19)
    x51 = np.zeros((nocc, nvir), dtype=np.float64)
    x51 -= einsum("ia->ia", x50)
    del x50
    rdm2_f_oooo += einsum("ij,kl->jikl", delta_oo, x19) * 4
    rdm2_f_oooo -= einsum("ij,kl->jlki", delta_oo, x19) * 2
    rdm2_f_oooo -= einsum("ij,kl->kjil", delta_oo, x19) * 2
    rdm2_f_oooo += einsum("ij,kl->klij", delta_oo, x19) * 4
    del x19
    x25 = np.zeros((nocc, nvir), dtype=np.float64)
    x25 -= einsum("ia->ia", x18)
    rdm2_f_oovo -= einsum("ij,ka->jiak", delta_oo, x18) * 4
    rdm2_f_oovo += einsum("ij,ka->jkai", delta_oo, x18) * 2
    del x18
    x20 = 0
    x20 += einsum("ai,ia->", l1a, t1a)
    x22 = 0
    x22 += x20
    del x20
    x21 = 0
    x21 += einsum("ai,ia->", l1a, t1b)
    x22 -= x21
    del x21
    x23 += einsum("ij->ji", delta_oo) * 2*x22
    x89 -= einsum("ia->ia", t1b) * 2*x22
    x90 += einsum("ia->ia", t1b) * 2*x22
    x23 -= einsum("ij->ji", delta_oo)
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x23) * 2
    rdm2_f_oooo -= einsum("ij,kl->lkji", delta_oo, x23) * 4
    rdm2_f_ooov -= einsum("ia,jk->kjia", t1b, x23) * 4
    rdm2_f_ooov += einsum("ia,jk->ijka", t1b, x23) * 2
    rdm2_f_ovoo -= einsum("ia,jk->iakj", t1b, x23) * 4
    rdm2_f_ovoo += einsum("ia,jk->kaij", t1b, x23) * 2
    del x23
    x25 += einsum("ai->ia", l1a)
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum("ia,jkab->ijkb", x25, t2b)
    x27 += einsum("ijka->ikja", x26)
    rdm2_f_ooov += einsum("ijka->jika", x27) * 2
    rdm2_f_ooov -= einsum("ijka->kija", x27) * 4
    rdm2_f_ovoo -= einsum("ijka->jaki", x27) * 4
    rdm2_f_ovoo += einsum("ijka->kaji", x27) * 2
    del x27
    x53 += einsum("ijka->ikja", x26)
    del x26
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x54 += einsum("ia,ijkb->jkab", t1b, x53)
    del x53
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x56 += einsum("ijab->ijab", x54)
    del x54
    x37 = np.zeros((nocc, nocc), dtype=np.float64)
    x37 += einsum("ia,ja->ij", t1b, x25)
    x38 = np.zeros((nocc, nvir), dtype=np.float64)
    x38 += einsum("ia,ji->ja", t1b, x37)
    x39 += einsum("ia->ia", x38)
    del x38
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x55 += einsum("ij,jkab->ikab", x37, t2b)
    del x37
    x56 += einsum("ijab->ijba", x55)
    del x55
    rdm2_f_ovov += einsum("ijab->iajb", x56) * 2
    rdm2_f_ovov -= einsum("ijab->ibja", x56) * 4
    rdm2_f_ovov -= einsum("ijab->jaib", x56) * 4
    rdm2_f_ovov += einsum("ijab->jbia", x56) * 2
    del x56
    x105 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x105 += einsum("ia,ijbc->jabc", x25, t2b)
    x107 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x107 += einsum("iabc->iacb", x105)
    del x105
    x106 = np.zeros((nvir, nvir), dtype=np.float64)
    x106 += einsum("ia,ib->ab", t1b, x25)
    x107 += einsum("ia,bc->icab", t1b, x106)
    del x106
    rdm2_f_vooo -= einsum("ij,ka->aijk", delta_oo, x25) * 2
    rdm2_f_vooo += einsum("ij,ka->akji", delta_oo, x25) * 4
    rdm2_f_oovv -= einsum("ia,jb->ijba", t1b, x25) * 2
    rdm2_f_ovvo += einsum("ia,jb->iabj", t1b, x25) * 4
    rdm2_f_voov += einsum("ia,jb->bjia", t1b, x25) * 4
    rdm2_f_vvoo -= einsum("ia,jb->baij", t1b, x25) * 2
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x35 -= einsum("ijab->jiab", t2b)
    x35 += einsum("ijab->jiba", t2b) * 2
    x36 = np.zeros((nocc, nvir), dtype=np.float64)
    x36 += einsum("ia,ijab->jb", x25, x35)
    del x25
    x39 -= einsum("ia->ia", x36)
    rdm2_f_ooov -= einsum("ij,ka->jika", delta_oo, x39) * 4
    rdm2_f_ooov += einsum("ij,ka->kjia", delta_oo, x39) * 2
    rdm2_f_ovoo += einsum("ij,ka->jaki", delta_oo, x39) * 2
    rdm2_f_ovoo -= einsum("ij,ka->kaij", delta_oo, x39) * 4
    del x39
    x51 += einsum("ia->ia", x36)
    del x36
    x52 += einsum("ia,jb->ijab", t1b, x51)
    del x51
    rdm2_f_ovov += einsum("ijab->iajb", x52) * 4
    rdm2_f_ovov -= einsum("ijab->ibja", x52) * 2
    rdm2_f_ovov -= einsum("ijab->jaib", x52) * 2
    rdm2_f_ovov += einsum("ijab->jbia", x52) * 4
    del x52
    rdm2_f_ovov -= einsum("ijab->jbia", x35) * 4*x22
    del x22
    rdm2_f_oovv -= einsum("abij,ikcb->kjac", l2a, x35) * 2
    rdm2_f_voov += einsum("ijab,ikac->bjkc", x11, x35) * 2
    del x35
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x58 += einsum("abij,kjbc->ikac", l2a, t2b)
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x59 += einsum("ijab,jkac->ikbc", t2b, x58)
    x64 += einsum("ijab->ijab", x59)
    del x59
    x109 += einsum("ijab->ijab", x58)
    del x58
    x110 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x110 += einsum("ia,ijbc->jabc", t1b, x109)
    del x109
    rdm2_f_ovvv += einsum("iabc->iabc", x110) * 2
    rdm2_f_ovvv -= einsum("iabc->icba", x110) * 4
    rdm2_f_vvov -= einsum("iabc->baic", x110) * 4
    rdm2_f_vvov += einsum("iabc->bcia", x110) * 2
    del x110
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x63 += einsum("ijab->jiba", t2b)
    x63 += einsum("ia,jb->ijab", t1b, t1b)
    x64 -= einsum("ijab->jiba", x63) * 2*x13
    del x13
    del x63
    rdm2_f_ovov += einsum("ijab->iajb", x64) * 4
    rdm2_f_ovov -= einsum("ijab->ibja", x64) * 2
    del x64
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x65 += einsum("abij,kjac->ikbc", l2a, t2b)
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x66 += einsum("ijab,jkac->ikbc", t2b, x65)
    rdm2_f_ovov -= einsum("ijab->iajb", x66) * 2
    rdm2_f_ovov += einsum("ijab->ibja", x66) * 4
    del x66
    x104 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x104 += einsum("ia,ijbc->jbac", t1b, x65)
    del x65
    x107 -= einsum("iabc->iabc", x104)
    del x104
    rdm2_f_ovvv += einsum("iabc->ibac", x107) * 4
    rdm2_f_ovvv -= einsum("iabc->icab", x107) * 2
    rdm2_f_vvov -= einsum("iabc->abic", x107) * 2
    rdm2_f_vvov += einsum("iabc->acib", x107) * 4
    del x107
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x68 += einsum("ijab->jiba", t2b)
    x68 += einsum("ia,jb->ijab", t1b, t1b)
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x69 += einsum("ijkl,ijab->klab", x0, x68) * 4
    del x0
    del x68
    x78 += einsum("ijab->jiba", x69) * -1
    del x69
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x70 += einsum("ijab->jiba", t2a)
    x70 += einsum("ijab->jiba", t2b) * -1
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x71 += einsum("ijab->ijba", x70)
    x71 += einsum("ijab->ijab", x70) * -0.5
    x85 = 0
    x85 += einsum("abij,ijba->", l2a, x70)
    rdm2_f_ovov += einsum("ia,jb->iajb", t1b, t1b) * 4.0*x85
    rdm2_f_ovov += einsum("ia,jb->jaib", t1b, t1b) * -2.0*x85
    del x85
    x88 = 0
    x88 += einsum("abij,ijab->", l2a, x70)
    del x70
    rdm2_f_ovov += einsum("ia,jb->iajb", t1b, t1b) * -8.0*x88
    rdm2_f_ovov += einsum("ia,jb->jaib", t1b, t1b) * 4.0*x88
    del x88
    x71 += einsum("ia,jb->ijab", t1a, t1a) * 0.5000000000000003
    x71 += einsum("ia,jb->ijab", t1b, t1b) * 0.5000000000000003
    x72 = 0
    x72 += einsum("abij,ijba->", l2a, x71)
    del x71
    x78 += einsum("ijab->jiba", t2b) * 8.0*x72
    del x72
    x73 = np.zeros((nocc, nvir), dtype=np.float64)
    x73 += einsum("ia,abji->jb", t1a, l2a)
    x74 = 0
    x74 += einsum("ia,ia->", t1a, x73)
    del x73
    x77 = 0
    x77 += 1.0*x74
    del x74
    x75 = np.zeros((nocc, nvir), dtype=np.float64)
    x75 += einsum("ia,baij->jb", t1b, l2a)
    x76 = 0
    x76 += einsum("ia,ia->", t1b, x75)
    del x75
    x77 += 1.0*x76
    del x76
    x78 += einsum("ia,jb->ijab", t1b, t1b) * 4.0*x77
    del x77
    rdm2_f_ovov += einsum("ijab->iajb", x78) * -1
    rdm2_f_ovov += einsum("ijab->ibja", x78) * 0.5
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x79 += einsum("abij->jiab", l2a) * -1
    x79 += einsum("abij->jiba", l2a) * 2
    x80 = np.zeros((nvir, nvir), dtype=np.float64)
    x80 += einsum("ijab,ijac->cb", t2b, x79) * 0.5
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x81 += einsum("ab,ijac->ijbc", x80, t2b) * 8
    x84 += einsum("ijab->jiba", x81)
    del x81
    rdm2_f_ovov += einsum("ijab->iajb", x84) * -1
    rdm2_f_ovov += einsum("ijab->ibja", x84) * 0.5
    rdm2_f_ovov += einsum("ijab->jaib", x84) * 0.5
    rdm2_f_ovov += einsum("ijab->jbia", x84) * -1
    del x84
    x97 = np.zeros((nvir, nvir), dtype=np.float64)
    x97 += einsum("ab->ab", x80)
    rdm2_f_ovvv += einsum("ia,bc->iabc", t1b, x80) * 8
    rdm2_f_ovvv += einsum("ia,bc->icba", t1b, x80) * -4
    rdm2_f_vvov += einsum("ia,bc->baic", t1b, x80) * -4
    rdm2_f_vvov += einsum("ia,bc->bcia", t1b, x80) * 8
    del x80
    x100 = np.zeros((nvir, nvir), dtype=np.float64)
    x100 += einsum("ijab,ijac->cb", t2b, x79)
    del x79
    x102 = np.zeros((nvir, nvir), dtype=np.float64)
    x102 += einsum("ab->ab", x100)
    del x100
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x87 += einsum("ijab->jiba", t2b)
    x87 += einsum("ia,jb->ijab", t1b, t1b)
    rdm2_f_ovov += einsum("ijab->jbia", x87) * 8.0*x86
    rdm2_f_ovov += einsum("ijab->jaib", x87) * -4.0*x86
    del x86
    del x87
    x89 += einsum("ia->ia", t1b)
    rdm2_f_ovov += einsum("ia,jb->jbia", t1b, x89) * 4
    del x89
    x90 -= einsum("ia->ia", t1b)
    rdm2_f_ovov += einsum("ia,jb->ibja", t1b, x90) * 2
    del x90
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x91 += einsum("ijab->jiab", t2b) * 2
    x91 -= einsum("ijab->jiba", t2b)
    rdm2_f_oovv -= einsum("abij,ikca->kjbc", l2a, x91) * 2
    rdm2_f_ovvo += einsum("ijab,ikca->kcbj", x11, x91) * 2
    del x91
    del x11
    x93 = np.zeros((nocc, nvir), dtype=np.float64)
    x93 += einsum("ia->ia", t1a)
    x93 += einsum("ia->ia", t1b) * -1
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x94 += einsum("abij->jiab", l2a) * -0.5
    x94 += einsum("abij->jiba", l2a)
    x95 = np.zeros((nocc, nvir), dtype=np.float64)
    x95 += einsum("ia,ijab->jb", x93, x94) * 2
    del x93
    del x94
    x96 = np.zeros((nocc, nvir), dtype=np.float64)
    x96 += einsum("ia->ia", x95) * -1
    del x95
    x96 += einsum("ai->ia", l1a)
    x97 += einsum("ia,ib->ba", t1b, x96) * 0.5
    rdm2_f_oovv += einsum("ij,ab->jiab", delta_oo, x97) * 8
    del x97
    x101 = np.zeros((nvir, nvir), dtype=np.float64)
    x101 += einsum("ia,ib->ab", t1b, x96)
    del x96
    x102 += einsum("ab->ba", x101)
    del x101
    rdm2_f_ovvo += einsum("ij,ab->jbai", delta_oo, x102) * -2
    rdm2_f_voov += einsum("ij,ab->aijb", delta_oo, x102) * -2
    rdm2_f_vvoo += einsum("ij,ab->abji", delta_oo, x102) * 4
    del x102
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x103 += einsum("abij->jiab", l2a) * 2
    x103 -= einsum("abij->jiba", l2a)
    rdm2_f_vvoo -= einsum("ijab,ikbc->cajk", t2b, x103) * 2
    del x103
    x112 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x112 += einsum("ia,bcji->jbca", t1b, l2a)
    x114 = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    x114 += einsum("ia,ibcd->cbda", t1b, x112)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_vvvv -= einsum("abcd->bcad", x114) * 2
    rdm2_f_vvvv += einsum("abcd->bdac", x114) * 4
    del x114
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_vovv += einsum("iabc->aibc", x112) * 4
    rdm2_f_vovv -= einsum("iabc->biac", x112) * 2
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_vvvo -= einsum("iabc->acbi", x112) * 2
    rdm2_f_vvvo += einsum("iabc->bcai", x112) * 4
    del x112
    x113 = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    x113 += einsum("abij,ijcd->abcd", l2a, t2b)
    rdm2_f_vvvv += einsum("abcd->bcad", x113) * -2
    rdm2_f_vvvv += einsum("abcd->bdac", x113) * 4
    del x113
    rdm2_f_oovo += einsum("ij,ak->jiak", delta_oo, l1a) * 4
    rdm2_f_oovo -= einsum("ij,ak->jkai", delta_oo, l1a) * 2
    rdm2_f_ovov -= einsum("ijab->jaib", t2b) * 2
    rdm2_f_ovov += einsum("ijab->jbia", t2b) * 4
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_vovo -= einsum("abij->ajbi", l2a) * 2
    rdm2_f_vovo += einsum("abij->bjai", l2a) * 4

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    return rdm2_f

