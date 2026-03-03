#!/usr/bin/env node
/**
 * DeepSeek PoW Solver — fast Node.js implementation
 * Called by deepseek_cli.py as subprocess.
 *
 * Usage: node pow_solver.js <challenge> <salt> <difficulty> <expire_at>
 * Output: JSON {"answer": N} or {"error": "..."}
 */

const RC32 = new Uint32Array([
    0, 1, 0, 32898, 0x80000000, 32906, 0x80000000, 0x80008000,
    0, 32907, 0, 0x80000001, 0x80000000, 0x80008081, 0x80000000, 32777,
    0, 138, 0, 136, 0, 0x80008009, 0, 0x8000000a,
    0, 0x8000808b, 0x80000000, 139, 0x80000000, 32905, 0x80000000, 32771,
    0x80000000, 32770, 0x80000000, 128, 0, 32778, 0x80000000, 0x8000000a,
    0x80000000, 0x80008081, 0x80000000, 32896, 0, 0x80000001, 0x80000000, 0x80008008,
]);
const V = [10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1];
const W_ROT = [1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44];

function absorb32(q, s) {
    for (let r = 0; r < q.length; r += 8) {
        const n = r >> 2;
        s[n] ^= q[r+7]<<24 | q[r+6]<<16 | q[r+5]<<8 | q[r+4];
        s[n+1] ^= q[r+3]<<24 | q[r+2]<<16 | q[r+1]<<8 | q[r];
    }
}

function squeeze32(s, b) {
    for (let r = 0; r < b.length; r += 8) {
        const n = r >> 2;
        b[r]=s[n+1]; b[r+1]=s[n+1]>>>8; b[r+2]=s[n+1]>>>16; b[r+3]=s[n+1]>>>24;
        b[r+4]=s[n]; b[r+5]=s[n]>>>8; b[r+6]=s[n]>>>16; b[r+7]=s[n]>>>24;
    }
}

function keccakF(A) {
    const C = new Int32Array(10);
    for (let ri = 1; ri < 24; ri++) {
        for (let t = 0; t < 5; t++) {
            const n2 = 2*t;
            C[n2] = A[n2]^A[n2+10]^A[n2+20]^A[n2+30]^A[n2+40];
            C[n2+1] = A[n2+1]^A[n2+11]^A[n2+21]^A[n2+31]^A[n2+41];
        }
        for (let t = 0; t < 5; t++) {
            const ci = ((t+1)%5)*2;
            const o = C[ci], f = C[ci+1];
            const d0 = C[((t+4)%5)*2] ^ ((o<<1)|(f>>>31));
            const d1 = C[((t+4)%5)*2+1] ^ ((f<<1)|(o>>>31));
            for (let r = 0; r < 25; r += 5) {
                const idx = (r+t)*2;
                A[idx] ^= d0; A[idx+1] ^= d1;
            }
        }
        let w0 = A[2], w1 = A[3];
        for (let ii = 0; ii < 24; ii++) {
            const tIdx = V[ii], aVal = W_ROT[ii];
            const c0 = A[2*tIdx], c1 = A[2*tIdx+1];
            const aMod = aVal & 31, sMod = (32 - aVal) & 31;
            const v0 = (w0 << aMod) | (w1 >>> sMod);
            const v1 = (w1 << aMod) | (w0 >>> sMod);
            if (aVal < 32) { w0 = v0; w1 = v1; }
            else { w0 = v1; w1 = v0; }
            A[2*tIdx] = w0; A[2*tIdx+1] = w1;
            w0 = c0; w1 = c1;
        }
        for (let t = 0; t < 25; t += 5) {
            for (let n = 0; n < 5; n++) {
                C[2*n] = A[(t+n)*2]; C[2*n+1] = A[(t+n)*2+1];
            }
            for (let n = 0; n < 5; n++) {
                const idx = (t+n)*2;
                A[idx] ^= ~C[((n+1)%5)*2] & C[((n+2)%5)*2];
                A[idx+1] ^= ~C[((n+1)%5)*2+1] & C[((n+2)%5)*2+1];
            }
        }
        A[0] ^= RC32[2*ri]; A[1] ^= RC32[2*ri+1];
    }
}

class DSKeccak {
    constructor() {
        this.state = new Int32Array(50);
        this.queue = Buffer.alloc(136);
        this.qoff = 0;
    }
    update(str) {
        const data = Buffer.from(str, 'utf8');
        for (let i = 0; i < data.length; i++) {
            this.queue[this.qoff] = data[i];
            this.qoff++;
            if (this.qoff >= 136) {
                absorb32(this.queue, this.state);
                keccakF(this.state);
                this.qoff = 0;
            }
        }
        return this;
    }
    digest() {
        const st = new Int32Array(this.state);
        const q = Buffer.alloc(136);
        this.queue.copy(q);
        q.fill(0, this.qoff);
        q[this.qoff] |= 6;
        q[135] |= 0x80;
        absorb32(q, st);
        keccakF(st);
        const buf = Buffer.alloc(32);
        squeeze32(st, buf);
        return buf.toString('hex');
    }
    copy() {
        const k = new DSKeccak();
        k.state = new Int32Array(this.state);
        this.queue.copy(k.queue);
        k.qoff = this.qoff;
        return k;
    }
}

// Main solver
const [,, challenge, salt, difficulty, expireAt] = process.argv;
if (!challenge || !salt || !difficulty || !expireAt) {
    console.error('Usage: node pow_solver.js <challenge> <salt> <difficulty> <expire_at>');
    process.exit(1);
}

const diff = parseInt(difficulty);
const prefix = `${salt}_${expireAt}_`;
const base = new DSKeccak();
base.update(prefix);

for (let i = 0; i < diff * 2; i++) {
    if (base.copy().update(String(i)).digest() === challenge) {
        console.log(JSON.stringify({ answer: i }));
        process.exit(0);
    }
}
console.log(JSON.stringify({ error: "no solution found" }));
process.exit(1);
