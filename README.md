# Polann

![GitHub License](https://img.shields.io/github/license/x-vmaier/polann?style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/x-vmaier/polann?style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/x-vmaier/polann/:workflow?style=flat-square)
![GitHub Release](https://img.shields.io/github/v/release/x-vmaier/polann?include_prereleases&style=flat-square)
![Development Status](https://img.shields.io/badge/status-alpha-red?style=flat-square)

**Polann** (Performance-oriented library for artificial neural networks) is a C++ library for building, training, and running neural networks.

## Motivation

After getting into the basics of neural networks, I wanted to try building one myself. Once I had a small network of dense layers with random weights and biases running, I got hooked on the idea of making it faster and maybe even usable for real-world applications. That's how **Polann** was born.

This project is still experimental and not yet production-ready.

## Roadmap

- [x] Basic dense layers implemented
- [x] Forward propagation
- [x] Backpropagation
- [ ] OpenBLAS acceleration for matrix operations
- [ ] Quantization (QAT, PTQ)
- [ ] Saving and loading models
- [ ] Convolutional layers
- [ ] Recurrent layers

## Building from source

```bash
git clone https://github.com/x-vmaier/polann
cd polann
cmake -S . -B build
cmake --build build
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
