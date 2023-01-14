import torch

from vsa_codebook_decoder.model.binder import FourierBinder

from vsa_codebook_decoder.codebook import vsa, codebook, Codebook


class TestBinder:
    def test_forward(self):
        n_features = 5
        latent_dim = 1024
        hd_placeholders = [vsa.generate(latent_dim) for i in range(n_features)]
        hd_placeholders = torch.stack(hd_placeholders)
        assert hd_placeholders.shape == (n_features, latent_dim)

        binder = FourierBinder(placeholders=hd_placeholders)
        batch_size = 10
        batch = torch.randn((batch_size, n_features, latent_dim))

        result_batch = torch.zeros_like(batch)
        for i in range(batch_size):
            for j in range(n_features):
                result_batch[i,j] = vsa.bind(batch[i,j], hd_placeholders[j])

        print(result_batch)
        print(binder(batch))

        assert torch.allclose(binder(batch),result_batch, atol=1e-5)
