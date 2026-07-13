# CharDataLoader (examples/char_lm_pc.jl) — C-03, port of dataloader.py's
# _TokenSequenceLoader/CharDataLoader. Offline: exercises the split/vocab/windowing logic
# against small synthetic corpora, no network fetch (the real download is smoke-tested by
# running the example directly, not by this suite).
using Test
using Random

include(joinpath(@__DIR__, "..", "examples", "char_lm_pc.jl"))

@testset "char_lm_pc.jl: Tiny Shakespeare loader (C-03)" begin
    @testset "_tiny_shakespeare_splits: TFDS 90/5/5 formula (tiny_shakespeare_dataset_builder.py:52-56)" begin
        text = "abcdefghijklmnopqrst"                      # 20 chars, all distinct
        train, validation, test = _tiny_shakespeare_splits(text)
        @test train == "abcdefghijklmnopqr"                # floor(20*0.9) = 18
        @test validation == "s"                             # floor(2*0.5) = 1
        @test test == "t"
        @test train * validation * test == text             # lossless partition, in order

        # a length where the 90% cut lands on a non-integer boundary (int() truncates)
        text2 = "0123456789"                                # 10 chars
        t2, v2, te2 = _tiny_shakespeare_splits(text2)
        @test t2 == "012345678"                              # floor(10*0.9) = 9
        @test v2 == ""                                       # floor(1*0.5) = 0
        @test te2 == "9"
        @test t2 * v2 * te2 == text2
    end

    @testset "_char_vocab: sorted-unique, 1-based (DIVERGENCE from upstream's 0-based)" begin
        chars, char_to_idx, idx_to_char = _char_vocab("banana")
        @test chars == ['a', 'b', 'n']
        @test char_to_idx == Dict('a' => 1, 'b' => 2, 'n' => 3)
        @test idx_to_char == Dict(1 => 'a', 2 => 'b', 3 => 'n')
    end

    @testset "CharDataLoader: sliding window (x, y = x shifted by one), drop-incomplete-batch" begin
        corpus = "abcdefghij"                                # 10 distinct, alphabetical ⇒ id == position
        chars, c2i, i2c = _char_vocab(corpus)
        @test length(chars) == 10
        seq_len, batch_size = 3, 2
        loader = CharDataLoader(
            corpus, chars, c2i, i2c, seq_len, batch_size; shuffle=false
        )
        @test loader.num_sequences == length(corpus) - seq_len            # 10 - 3 = 7
        @test length(loader) == 7 ÷ 2                                     # 3 full batches, 1 dropped

        batches = collect(loader)
        @test length(batches) == length(loader)
        # unshuffled ⇒ deterministic window starts 1,2 | 3,4 | 5,6 (start 7 dropped, incomplete)
        @test batches[1]["x"] == Float32[1 2 3; 2 3 4]
        @test batches[1]["y"] == Float32[2 3 4; 3 4 5]
        # general sliding-window invariant, independent of exact ids: y[1:S-1] == x[2:S]
        for b in batches, row in axes(b["x"], 1)
            @test b["y"][row, 1:(end - 1)] == b["x"][row, 2:end]
        end

        # max_samples caps num_sequences (and therefore num_batches)
        capped = CharDataLoader(
            corpus, chars, c2i, i2c, seq_len, batch_size; shuffle=false, max_samples=3
        )
        @test capped.num_sequences == 3
        @test length(capped) == 3 ÷ 2
    end

    @testset "CharDataLoader: re-iterable across epochs, seeded shuffle is reproducible" begin
        corpus = "abcdefghijklmnopqrstuvwxyz0123456789"       # 36 distinct chars
        chars, c2i, i2c = _char_vocab(corpus)
        seq_len, batch_size = 4, 3
        la = CharDataLoader(
            corpus, chars, c2i, i2c, seq_len, batch_size; shuffle=true, seed=42
        )
        lb = CharDataLoader(
            corpus, chars, c2i, i2c, seq_len, batch_size; shuffle=true, seed=42
        )
        @test collect(la) == collect(lb)                      # same seed ⇒ identical epoch-1 order

        # `la` has now been iterated once (epoch 1 consumed) -- a second `for batch in la`
        # (epoch 2) must still yield valid, shift-by-one windows (reshuffled, not stale state).
        epoch2 = collect(la)
        @test length(epoch2) == length(la)
        for b in epoch2, row in axes(b["x"], 1)
            @test b["y"][row, 1:(end - 1)] == b["x"][row, 2:end]
        end
    end

    @testset "decode: char-index round-trip" begin
        corpus = "hello world"
        chars, c2i, i2c = _char_vocab(corpus)
        loader = CharDataLoader(corpus, chars, c2i, i2c, 3, 1; shuffle=false)
        idxs = [c2i[ch] for ch in "hello"]
        @test decode(loader, idxs) == "hello"
    end
end
