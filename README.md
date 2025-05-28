# rfm-customer-segmentation
# 🧠 Customer Segmentation using RFM and Clustering

This project analyzes customer behavior based on Recency, Frequency, and Monetary value (RFM), and applies clustering algorithms to segment customers for targeted marketing strategies.

## 🔗 Run Online

You can run this project interactively on Google Colab:  
[Open in Google Colab](https://colab.research.google.com/drive/1dH3XHSSzvBT9WSs-wa1khsRp9Wbl8Onx?usp=sharing)

## 📊 Features
- RFM Analysis to quantify customer engagement  
- Clustering with KMeans, Hierarchical Clustering, and DBSCAN  
- Model validation using Davies-Bouldin Score & Elbow Method  
- Visualizations: distribution plots, pairplots, dendrograms, 2D & 3D cluster views  
- Business recommendations tailored for each customer segment  

## 📁 Folder Structure
- `data/`: Original dataset (`Online Retail.xlsx`)  
- `notebooks/`: Jupyter or Google Colab notebooks with analysis  
- `images/`: Exported visualizations and plots  
- `src/`: (Optional) Python scripts for clustering pipeline  

## 🚀 How to Run
1. Install project dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the Jupyter notebook:
    - Open `notebooks/customer_segmentation.ipynb` in Jupyter or Google Colab  
    - Execute all cells sequentially  

3. (Optional) Run clustering script:
    ```bash
    python src/rfm_clustering.py
    ```

## 📎 Dataset
The dataset `Online Retail.xlsx` is included in the `data/` folder.  
You can download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) or use your own transactional retail dataset.  

---

If you have questions or want to contribute, feel free to open issues or pull requests.

---

*Happy Clustering!* 🚀
