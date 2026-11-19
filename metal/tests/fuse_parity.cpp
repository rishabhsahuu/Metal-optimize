// metal/tests/fuse_parity.cpp - the decode latency fusions against the kernels they replace, BITWISE, for 1..4
// tokens:
//   gemv_cat (gemv.metal)   ssm_alpha + ssm_beta (BF16 2560 -> 48 + 48) in one dispatch  ==  two gemv dispatches
//   router_route            the router GEMV + moe_route in one dispatch  ==  gemv + moe_route (logits, ids, w, plan;
//                           twice in a row: the counter resets)
//   gr_norm_comb (gr2.metal) the MoE combine folded into the next gr_norm  ==  moe_combine_sh + gr_norm (R', xn)
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
constexpr int P_INTS = 16 + 128 + 128 + 4 + 128 * 3, P_TOKENT = 16 + 128 + 128 + 4 + 128 * 2;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct HcfArgs { uint32_t n_embd, hc; float eps; uint32_t pad; };
struct GlueArgs { uint32_t n, n_tok, n_embd, topk; };
}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal", "gemv.metal", "router.metal", "gr.metal",
                                "gr2.metal", "glue.metal", "hcf.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    int fails = 0;
    const int N = 2560, NO = 48, MAXT = 8;
    const size_t rb = ref::row_bytes(ref::BF16, N);
    mtl::Buffer Wa = cx.alloc(rb * NO), Wb = cx.alloc(rb * NO), xp = cx.alloc(4 * ((size_t)4 * MAXT * N + MAXT * N / 16)),
                y1 = cx.alloc(4 * MAXT * NO), y2 = cx.alloc(4 * MAXT * NO), z1 = cx.alloc(4 * MAXT * NO), z2 = cx.alloc(4 * MAXT * NO);
    ref::random_blocks(ref::BF16, Wa.as<uint8_t>(), NO, N, 1, 0.05f);
    ref::random_blocks(ref::BF16, Wb.as<uint8_t>(), NO, N, 2, 0.05f);
    ref::Rng r(3);
    for (size_t i = 0; i < xp.bytes / 4; ++i) xp.as<float>()[i] = r.sym();
    for (int nt = 1; nt <= MAXT; ++nt) {   // every decode window size the engine builds
        mtl::Pipeline pg = cx.pipeline(lib, "gemv_30_32_1_4_1", mtl::Constants().i(0, nt)),
                      pc = cx.pipeline(lib, "gemvc_30_32_1_4", mtl::Constants().i(0, nt));
        const size_t s16 = (size_t)4 * 4 * MAXT * N;
        GemvArgs a{(uint32_t)N, (uint32_t)NO, (uint32_t)nt, (uint32_t)rb, (uint32_t)N, (uint32_t)NO};
        auto g = [&](const mtl::Buffer& W, const mtl::Buffer& y) {
            mtl::Dispatch d;
            d.p = &pg;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(y), mtl::Arg::V(a)};
            d.grid[0] = NO / 4;
            d.tg[0] = 128;
            return d;
        };
        cx.run({g(Wa, y1), g(Wb, y2)});
        GemvArgs ac = a;
        ac.n_out = 2 * NO;
        const uint32_t sp = NO;
        mtl::Dispatch d;
        d.p = &pc;
        d.args = {mtl::Arg::B(Wa), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(z1), mtl::Arg::V(ac), mtl::Arg::B(Wb),
                  mtl::Arg::B(z2), mtl::Arg::V(sp)};
        d.grid[0] = 2 * NO / 4;
        d.tg[0] = 128;
        cx.run({d});
        const bool ok = std::memcmp(y1.ptr, z1.ptr, 4 * nt * NO) == 0 && std::memcmp(y2.ptr, z2.ptr, 4 * nt * NO) == 0;
        fails += !ok;
        std::printf("gemv_cat (alpha + beta) ntok %d: bitwise %s\n", nt, ok ? "PASS" : "FAIL");
    }
    // ---- router_route
    {
        for (int rty : {ref::BF16, ref::F32})
        for (const int NE : {512, 256}) {   // (256: the expert-pruned Coder; the logits rows keep their 512-float stride)
        const size_t rbr = ref::row_bytes(rty, N);
        mtl::Buffer Wr = cx.alloc(rbr * NE), lg1 = cx.alloc(4 * MAXT * 512), lg2 = cx.alloc(4 * MAXT * 512),
                    id1 = cx.alloc(4 * MAXT * 10), id2 = cx.alloc(4 * MAXT * 10), w1 = cx.alloc(4 * MAXT * 10),
                    w2 = cx.alloc(4 * MAXT * 10), pl1 = cx.alloc(4 * P_INTS), pl2 = cx.alloc(4 * P_INTS), cnt = cx.alloc(16);
        ref::random_blocks(rty, Wr.as<uint8_t>(), NE, N, 9, 0.05f);
        mtl::Pipeline pr = cx.pipeline(lib, "moe_route");
        for (int nt = 1; nt <= MAXT; ++nt) {
            mtl::Pipeline pg = cx.pipeline(lib, "gemv_" + std::to_string(rty) + "_32_1_4_1", mtl::Constants().i(0, nt)),
                          pf = cx.pipeline(lib, "router_route_" + std::to_string(rty), mtl::Constants().i(0, nt));
            const size_t s16 = (size_t)4 * 4 * MAXT * N;
            GemvArgs ga{(uint32_t)N, (uint32_t)NE, (uint32_t)nt, (uint32_t)rbr, (uint32_t)N, 512u};
            MoeArgs ma{};
            ma.n_tok = nt;
            ma.topk = 10;
            ma.n_expert = NE;
            ma.gx_gu = 5;
            ma.gx_dn = 20;
            mtl::Dispatch g;
            g.p = &pg;
            g.args = {mtl::Arg::B(Wr), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(lg1), mtl::Arg::V(ga)};
            g.grid[0] = NE / 4;
            g.tg[0] = 128;
            mtl::Dispatch rt;
            rt.p = &pr;
            rt.args = {mtl::Arg::B(lg1), mtl::Arg::B(id1), mtl::Arg::B(w1), mtl::Arg::B(pl1), mtl::Arg::V(ma)};
            rt.tg[0] = 512;
            std::memset(pl1.ptr, 0, pl1.bytes);
            cx.run({g, rt});
            bool ok = true;
            for (int rep_ = 0; rep_ < 2; ++rep_) {
                std::memset(pl2.ptr, 0, pl2.bytes);
                mtl::Dispatch f;
                f.p = &pf;
                f.args = {mtl::Arg::B(Wr), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(lg2), mtl::Arg::V(ga), mtl::Arg::B(id2),
                          mtl::Arg::B(w2), mtl::Arg::B(pl2), mtl::Arg::V(ma), mtl::Arg::B(cnt)};
                f.grid[0] = NE / 16;
                f.tg[0] = 512;
                cx.run({f});
                ok = ok && std::memcmp(lg1.ptr, lg2.ptr, 4 * nt * 512) == 0 && std::memcmp(id1.ptr, id2.ptr, 4 * nt * 10) == 0 &&
                     std::memcmp(w1.ptr, w2.ptr, 4 * nt * 10) == 0 && std::memcmp(pl1.ptr, pl2.ptr, 4 * P_INTS) == 0 &&
                     cnt.as<uint32_t>()[0] == 0;
            }
            fails += !ok;
            std::printf("router_route %s ntok %d: logits, ids, w, plan bitwise (2 runs) %s\n", ref::type_name(rty), nt,
                        ok ? "PASS" : "FAIL");
        }
        }
    }
    // ---- gr_norm_comb
    {
        const int HC = 4, TOPK = 10;
        mtl::Buffer R0 = cx.alloc(4 * MAXT * HC * N), R1 = cx.alloc(4 * MAXT * HC * N), R2 = cx.alloc(4 * MAXT * HC * N),
                    ye = cx.alloc(4 * (size_t)MAXT * TOPK * N), ysh = cx.alloc(4 * MAXT * N), sgv = cx.alloc(4 * MAXT),
                    plan = cx.alloc(4 * P_INTS), inj = cx.alloc(4 * MAXT * HC), wn = cx.alloc(4 * HC * N), bo = cx.alloc(4 * MAXT * N),
                    x1 = cx.alloc(4 * MAXT * HC * N), x2 = cx.alloc(4 * MAXT * HC * N);
        for (size_t i = 0; i < R0.bytes / 4; ++i) R0.as<float>()[i] = r.sym();
        for (size_t i = 0; i < ye.bytes / 4; ++i) ye.as<float>()[i] = r.sym();
        for (size_t i = 0; i < ysh.bytes / 4; ++i) ysh.as<float>()[i] = r.sym();
        for (int i = 0; i < MAXT; ++i) sgv.as<float>()[i] = r.sym() * 3;
        for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = r.sym() * 3;
        for (int i = 0; i < HC * N; ++i) wn.as<float>()[i] = 0.5f + r.uni();
        for (int i = 0; i < MAXT * TOPK; ++i) plan.as<int>()[P_TOKENT + i] = (i * 7) % (MAXT * TOPK);
        mtl::Pipeline pcm = cx.pipeline(lib, "moe_combine_sh"), pn = cx.pipeline(lib, "gr_norm"), pnc = cx.pipeline(lib, "gr_norm_comb");
        for (int nt = 1; nt <= MAXT; ++nt) {
            std::memcpy(R1.ptr, R0.ptr, R0.bytes);
            std::memcpy(R2.ptr, R0.ptr, R0.bytes);
            GlueArgs ga{(uint32_t)N, (uint32_t)nt, (uint32_t)N, TOPK};
            GrArgs gr{(uint32_t)N, HC, 320, (uint32_t)nt, 1, 1e-6f};
            mtl::Dispatch c; c.p = &pcm;
            c.args = {mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::B(ysh), mtl::Arg::B(sgv), mtl::Arg::B(bo), mtl::Arg::V(ga)};
            c.grid[0] = (nt * N + 255) / 256; c.tg[0] = 256;
            mtl::Dispatch n1; n1.p = &pn;
            n1.args = {mtl::Arg::B(R1), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(x1), mtl::Arg::V(gr)};
            n1.grid[0] = HC; n1.grid[1] = nt; n1.tg[0] = 256;
            cx.run({c, n1});
            const uint32_t tk = TOPK;
            mtl::Dispatch n2; n2.p = &pnc;
            n2.args = {mtl::Arg::B(R2), mtl::Arg::B(ye), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(x2), mtl::Arg::V(gr),
                       mtl::Arg::B(plan), mtl::Arg::B(ysh), mtl::Arg::B(sgv), mtl::Arg::V(tk)};
            n2.grid[0] = HC; n2.grid[1] = nt; n2.tg[0] = 256;
            cx.run({n2});
            const bool ok = std::memcmp(R1.ptr, R2.ptr, 4 * (size_t)nt * HC * N) == 0 && std::memcmp(x1.ptr, x2.ptr, 4 * (size_t)nt * HC * N) == 0;
            fails += !ok;
            std::printf("gr_norm_comb ntok %d: R', xn bitwise %s\n", nt, ok ? "PASS" : "FAIL");
        }
    }
    // ---- hcf (hcf.metal): the producer writes bo, applies it to R and its last threadgroup forms xn, bitwise
    //      gemv + gr_norm and moe_combine_sh + gr_norm
    {
        const int HC = 4, TOPK = 10;
        const size_t rbh = ref::row_bytes(ref::BF16, N);
        const size_t R0b = 4 * (size_t) MAXT * HC * N;
        mtl::Buffer Wh = cx.alloc(rbh * N), R0 = cx.alloc(R0b), R1 = cx.alloc(R0b), R2 = cx.alloc(R0b),
                    boh1 = cx.alloc(4 * MAXT * N), boh2 = cx.alloc(4 * MAXT * N), inj = cx.alloc(4 * MAXT * HC),
                    wn = cx.alloc(4 * HC * N), xn1 = cx.alloc(R0b), xn2 = cx.alloc(R0b), cnt = cx.alloc(16),
                    yer = cx.alloc(4 * (size_t) MAXT * TOPK * N), yshr = cx.alloc(4 * MAXT * N), sgv = cx.alloc(4 * MAXT),
                    plr = cx.alloc(4 * P_INTS);
        ref::random_blocks(ref::BF16, Wh.as<uint8_t>(), N, N, 11, 0.05f);
        for (size_t i = 0; i < R0b / 4; ++i) R0.as<float>()[i] = r.sym();
        for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = r.sym() * 3;
        for (int i = 0; i < HC * N; ++i) wn.as<float>()[i] = 0.5f + r.uni();
        for (size_t i = 0; i < yer.bytes / 4; ++i) yer.as<float>()[i] = r.sym();
        for (size_t i = 0; i < yshr.bytes / 4; ++i) yshr.as<float>()[i] = r.sym();
        for (int i = 0; i < MAXT; ++i) sgv.as<float>()[i] = r.sym() * 3;
        for (int i = 0; i < MAXT * TOPK; ++i) plr.as<int>()[P_TOKENT + i] = (i * 7) % (MAXT * TOPK);
        mtl::Pipeline pcm = cx.pipeline(lib, "moe_combine_sh"), pcm_h = cx.pipeline(lib, "moe_combine_hcf"),
                      pn = cx.pipeline(lib, "gr_norm");
        for (int nt = 1; nt <= MAXT; ++nt) {
            mtl::Pipeline pgv = cx.pipeline(lib, "gemv_30_32_1_8_2", mtl::Constants().i(0, nt)),
                          pgv_h = cx.pipeline(lib, "gemvh_30", mtl::Constants().i(0, nt));   // NTOK, as the engine sets it
            const size_t s16 = (size_t) 4 * 4 * MAXT * N;
            GemvArgs ga{(uint32_t) N, (uint32_t) N, (uint32_t) nt, (uint32_t) rbh, (uint32_t) N, (uint32_t) N};
            GrArgs gr{(uint32_t) N, HC, 320, (uint32_t) nt, 1, 1e-6f};
            HcfArgs hc{(uint32_t) N, HC, 1e-6f, 0};
            // gemv + gr_norm  ==  gemvh
            std::memcpy(R1.ptr, R0.ptr, R0b);
            std::memcpy(R2.ptr, R0.ptr, R0b);
            mtl::Dispatch g;
            g.p = &pgv;
            g.args = {mtl::Arg::B(Wh), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(boh1), mtl::Arg::V(ga)};
            g.grid[0] = N / 4;
            g.tg[0] = 256;
            mtl::Dispatch n1;
            n1.p = &pn;
            n1.args = {mtl::Arg::B(R1), mtl::Arg::B(boh1), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn1), mtl::Arg::V(gr)};
            n1.grid[0] = HC;
            n1.grid[1] = nt;
            n1.tg[0] = 256;
            cx.run({g, n1});
            std::memset(cnt.ptr, 0, 16);
            mtl::Dispatch g2;
            g2.p = &pgv_h;
            g2.args = {mtl::Arg::B(Wh), mtl::Arg::B(xp), mtl::Arg::B(xp, s16), mtl::Arg::B(boh2), mtl::Arg::V(ga),
                       mtl::Arg::B(R2), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn2), mtl::Arg::B(cnt), mtl::Arg::V(hc)};
            g2.grid[0] = N / 4;
            g2.tg[0] = 256;
            cx.run({g2});
            const bool b_ok = std::memcmp(boh1.ptr, boh2.ptr, 4 * (size_t) nt * N) == 0,
                       r_ok = std::memcmp(R1.ptr, R2.ptr, 4 * (size_t) nt * HC * N) == 0,
                       x_ok = std::memcmp(xn1.ptr, xn2.ptr, 4 * (size_t) nt * HC * N) == 0;
            bool ok = b_ok && r_ok && x_ok;
            if (!ok) {
                std::printf("  gemvh ntok %d: bo %d R %d xn %d\n", nt, (int) b_ok, (int) r_ok, (int) x_ok);
                for (int q = 0; q < nt * HC * N; ++q)
                    if (R1.as<float>()[q] != R2.as<float>()[q]) {
                        std::printf("    R first diff q %d (t %d c %d d %d): %.9g vs %.9g\n", q, q / (HC * N), (q / N) % HC, q % N,
                                    R1.as<float>()[q], R2.as<float>()[q]);
                        break;
                    }
            }
            // moe_combine_sh + gr_norm  ==  moe_combine_hcf
            std::memcpy(R1.ptr, R0.ptr, R0b);
            std::memcpy(R2.ptr, R0.ptr, R0b);
            GlueArgs cg{(uint32_t) N, (uint32_t) nt, (uint32_t) N, TOPK};
            mtl::Dispatch c;
            c.p = &pcm;
            c.args = {mtl::Arg::B(yer), mtl::Arg::B(plr), mtl::Arg::B(yshr), mtl::Arg::B(sgv), mtl::Arg::B(boh1), mtl::Arg::V(cg)};
            c.grid[0] = (uint32_t) ((nt * N + 255) / 256);
            c.tg[0] = 256;
            mtl::Dispatch n2;
            n2.p = &pn;
            n2.args = {mtl::Arg::B(R1), mtl::Arg::B(boh1), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn1), mtl::Arg::V(gr)};
            n2.grid[0] = HC;
            n2.grid[1] = nt;
            n2.tg[0] = 256;
            cx.run({c, n2});
            std::memset(cnt.ptr, 0, 16);
            mtl::Dispatch c2;
            c2.p = &pcm_h;
            c2.args = {mtl::Arg::B(yer), mtl::Arg::B(plr), mtl::Arg::B(yshr), mtl::Arg::B(sgv), mtl::Arg::B(boh2), mtl::Arg::V(cg),
                       mtl::Arg::B(R2), mtl::Arg::B(inj), mtl::Arg::B(wn), mtl::Arg::B(xn2), mtl::Arg::B(cnt), mtl::Arg::V(hc)};
            c2.grid[0] = (uint32_t) ((nt * N + 255) / 256);
            c2.tg[0] = 256;
            cx.run({c2});
            const bool b2 = std::memcmp(boh1.ptr, boh2.ptr, 4 * (size_t) nt * N) == 0,
                       r2 = std::memcmp(R1.ptr, R2.ptr, 4 * (size_t) nt * HC * N) == 0,
                       x2 = std::memcmp(xn1.ptr, xn2.ptr, 4 * (size_t) nt * HC * N) == 0;
            ok = ok && b2 && r2 && x2;
            if (!(b2 && r2 && x2)) std::printf("  combine ntok %d: bo %d R %d xn %d\n", nt, (int) b2, (int) r2, (int) x2);
            fails += !ok;
            std::printf("hcf gemvh / moe_combine_hcf ntok %d: bo, R', xn bitwise %s\n", nt, ok ? "PASS" : "FAIL");
        }
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
