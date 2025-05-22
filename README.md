<div align="center">
  <h1> 🔍 ZeroDINO</h1>
  <p><em>Semantic and Context-Aware Refinement with Adaptive Fusion for Generalized Zero-Shot Learning</em></p>
</div>

---

#  🧠 Model Architecture
![Model_architecture](framework/architecture.png)



# 🚀 Quick Start
Before you begin, please make sure you have downloaded the following datasets:

- 🐦 **[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)**
- 🌞 **[SUN Attribute](https://groups.csail.mit.edu/vision/SUN/hierarchy.html)**
- 🐘 **[AWA2](https://cvml.ist.ac.at/AwA2/)**


### ✅ 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### ✅ 2. Train or Evaluate 

Train on a specific dataset:

```bash
bash train.sh CUB      # or SUN / AWA2
```

Evaluate using pretrained weights:

```bash
bash test.sh CUB      # or SUN / AWA2
```

### 📌 Evaluate under CZSL setting:

Just add ```--zsl_only ``` to the test command.


# 📈 Results

Performance of our released models on three benchmark datasets under two evaluation protocols: Conventional Zero-Shot Learning (CZSL) and Generalized Zero-Shot Learning (GZSL).

| Dataset |Acc (CZSL)  | Unseen (GZSL) | Seen (GZSL) |Harmonic Mean (H)  |
|:-------:|:----------:|:-------------:|:-----------:|:-----------------:|
| **CUB** |    86.0    |     78.3      |    80.1     |       79.2        |
| **SUN** |    79.2    |     64.7      |    57.0     |       60.6        |
| **AWA2**|    67.5    |     64.5      |    87.9     |       74.4        |






