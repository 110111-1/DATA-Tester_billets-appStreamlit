import warnings
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_icon='📊', page_title="ONCFM", initial_sidebar_state="expanded"
)
st.sidebar.image('img/banniere.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.sidebar.write(
    """
# 📊 Tester les billets
Upload your file to test tickets
"""
)
col1: object
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("img/logo.jpg")

with col3:
    st.write(' ')


def plot_confusion_matrix(cf_matrix):
    plt.figure(figsize=(6, 6),dpi=40)
    group_names = ['True Negative', 'False Negative', 'False Positive', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title("Prédiction avec les données normalisées")
    plt.show()

def boucle_resultats(new):
    st.markdown("# Résultats")
    nb_True = 0
    nb_False = 0
    for j, i, k in zip(new.index, new["Prediction"], new["id"]):
        if i == True:
            nb_True += 1
            st.write("Le billet {}".format(k), "semble vrai avec une probabilité de "
                     + str(new.iloc[j, 9]), "%")
        else:
            nb_False += 1
            st.write("Le billet {}".format(k), "semble faux avec une probabilité de "
                     + str(new.iloc[j, 8]), "%")
    st.markdown("## Nombre de vrai billets : ")
    st.success(nb_True)
    st.markdown("## Nombre de faux billets : ")
    st.error(nb_False)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

use_example_file = st.checkbox(
    "Use example file", False, help="Utiliser les données exemples pour la démo"
)

if use_example_file:
    uploaded_file = "billets_production.csv"

if uploaded_file:
    new = pd.read_csv(uploaded_file)
    if st.sidebar.checkbox("Display data", False):
        st.subheader("Show dataset")
        st.markdown("### Data preview")
        st.dataframe(new.head())

    st.sidebar.subheader("Choose classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("reset", "Kmeans", "Logistic Regression", "Support Vector Machine (SVM)",  "Random Forest"))

    if classifier == "Logistic Regression":
        df = pd.read_csv("billets_complet.csv", sep=',')
        df.is_genuine = LabelEncoder().fit_transform(df['is_genuine'])
        X = df.drop(columns='is_genuine')
        y = df.is_genuine

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=42)
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        estimator = LogisticRegression()

        params = {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']
        }
        grid = GridSearchCV(estimator,
                            params,
                            cv=10,  # nb folds
                            n_jobs=-1,  # cpu
                            return_train_score=True,
                            verbose=1)
        grid.fit(X_train_scaled, y_train)
        best_params = grid.best_params_
        reg_log = LogisticRegression(**best_params)
        st.write(reg_log)
        reg_log.fit(X_train, y_train)
        y_pred = reg_log.predict(X_test)
        y_prob = reg_log.predict_proba(X_test).round(2)

        res = confusion_matrix(y_test, reg_log.predict(X_test))
        st.pyplot(plot_confusion_matrix(res))

        st.write('Precision: %.3f' % precision_score(y_test, y_pred))
        st.write('Recall   : %.3f' % recall_score(y_test, y_pred))
        st.write('Accuracy : %.3f' % accuracy_score(y_test, y_pred))
        st.write('F1 Score : %.3f' % f1_score(y_test, y_pred))
        X_new = new.drop(columns='id')
        y_pred_new = reg_log.predict(X_new)
        new["Prediction"] = y_pred_new
        proba_new = reg_log.predict_proba(X_new)
        new['Proba_Faux'] = proba_new[:, 0].round(4) * 100
        new['Proba_Vrai'] = proba_new[:, 1].round(4) * 100
        st.markdown("# Prédictions")
        st.markdown("## Tableau de résultat")
        st.dataframe(new.head())
        boucle_resultats(new)

    if classifier == "Kmeans":
        data = pd.read_csv("billets_complet.csv", sep=',')
        cluster_df = data
        X = data.values

        std_scale = preprocessing.StandardScaler().fit(X)
        X_scaled = std_scale.transform(X)

        km = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=150)
        km.fit(X_scaled)
        x_km = km.fit_transform(
            cluster_df[["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]])
        clusters_km = km.labels_
        cluster_df["cluster_km"] = km.labels_

        st.write("Les centroïdes du dataset d'entrainement")
        centroides_km = km.cluster_centers_
        st.write(centroides_km)
        pca_km = decomposition.PCA(n_components=3).fit(
            cluster_df[["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]])
        acp_km = PCA(n_components=3).fit_transform(
            cluster_df[["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]])
        centroides_km_projected = pca_km.transform(centroides_km)

        pca = decomposition.PCA(n_components=3).fit(X)
        X_projected = pca.transform(X)

        fig = plt.plot()
        plt.figsize = (20, 10)
        for couleur, k in zip(["#07ed31", "#ed070f"], [0, 1]):
            plt.scatter(acp_km[km.labels_ == k, 0], acp_km[km.labels_ == k, 1], c=couleur, edgecolors="indigo",
                        label="Cluster {}".format(k))
            plt.legend()
            plt.scatter(centroides_km_projected[:, 0], centroides_km_projected[:, 1], color="black", label="Centroïdes")
        plt.title(
            "Projection des individus et des {} centroïdes sur le premier plan factoriel".format(len(centroides_km)))
        st.pyplot(fig)

        # Modelisation
        df = pd.read_csv("billets_complet.csv", sep=',')
        df.is_genuine = LabelEncoder().fit_transform(df['is_genuine'])
        X = df.drop(columns='is_genuine')
        y = df.is_genuine
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=150)
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        modele = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=150)
        modele.fit(X_train, y_train)
        st.write("Centroides du model de prédiction")
        st.write(modele.cluster_centers_)
        y_pred = modele.predict(X_test)

        res = confusion_matrix(y_test, y_pred)
        st.pyplot(plot_confusion_matrix(res))

        st.write(f"Precision : {round(precision_score(y_test, modele.predict(X_test)) * 100, 2)} %")
        st.write(f"Recall    : {round(recall_score(y_test, modele.predict(X_test)) * 100, 2)} %")
        st.write(f"Accuracy  : {round(accuracy_score(y_test, modele.predict(X_test)) * 100, 2)} %")
        st.write(f"F1 score  : {round(f1_score(y_test, modele.predict(X_test)) * 100, 2)} %")

        # Test sur échantillon
        X_new = new.drop(columns='id')
        X_new = scaler.transform(X_new)

        y = modele.predict(X_new)
        new['Cluster'] = y

        st.markdown("### Résultats du test ")
        st.dataframe(new)

        nb_True = 0
        nb_False = 0
        for i, k in zip(y, new["id"]):
            if i == True:
                nb_True += 1
                st.write("Le billet {}".format(k), "semble vrai")
            else:
                nb_False += 1
                st.write("Le billet {}".format(k), "semble faux ")
        st.write("Nombre de vrai billets : ")
        st.success(nb_True)
        st.write("Nombre de faux billets : ")
        st.error(nb_False)

    if classifier == "Random Forest":
        data = pd.read_csv("billets_complet.csv", sep=',')
        y = data["is_genuine"]
        x = data.drop("is_genuine", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=42)
        modele_rf = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None, )
        modele_rf.fit(X_train, y_train)
        pd.DataFrame(modele_rf.feature_importances_,
                     index=X_train.columns,
                     columns=["importance"]).sort_values(
            "importance",
            ascending=False)
        st.write(f"Le pourcentage de bien classés est de : {accuracy_score(y_test, modele_rf.predict(X_test))*100} %")

        st.table(pd.DataFrame(confusion_matrix(y_test, modele_rf.predict(X_test)),
             index = ['faux données','vrai données'],
             columns = ['Faux predits','Vrai predit']))

        res = confusion_matrix(y_test, modele_rf.predict(X_test))
        st.pyplot(plot_confusion_matrix(res))
        st.markdown("# Predictions")
        X_new = new.drop(columns='id')
        modele_rf = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None, )
        modele_rf.fit(X_train, y_train)
        y_pred_new = modele_rf.predict(X_new)
        new["Prediction"] = y_pred_new
        proba_new = modele_rf.predict_proba(X_new)
        new['Proba_Faux'] = proba_new[:, 0].round(4) * 100
        new['Proba_Vrai'] = proba_new[:, 1].round(4) * 100

        st.markdown("## Tableau de résultats")
        st.table(new)
        boucle_resultats(new)

    if classifier == "Support Vector Machine (SVM)":
        data = pd.read_csv("billets_complet.csv", sep=',')
        y = data["is_genuine"]
        X = data.drop("is_genuine", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=42)
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        estimator = LinearSVC()
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)


        res = confusion_matrix(y_test, estimator.predict(X_test))
        st.pyplot(plot_confusion_matrix(res))

        st.write('Precision: %.3f' % precision_score(y_test, y_pred))
        st.write('Recall   : %.3f' % recall_score(y_test, y_pred))
        st.write('Accuracy : %.3f' % accuracy_score(y_test, y_pred))
        st.write('F1 Score : %.3f' % f1_score(y_test, y_pred))

        X_new = new.drop(columns='id')
        std_scale = preprocessing.StandardScaler().fit(X_new)
        X_scaled = std_scale.transform(X_new)
        estimator = LinearSVC()
        estimator.fit(X_train, y_train)
        y_pred_new = estimator.predict(X_new)
        new["Prediction"] = y_pred_new
        proba_new = estimator._predict_proba_lr(X_new)
        new['Proba_Faux'] = proba_new[:, 0].round(4) * 100
        new['Proba_Vrai'] = proba_new[:, 1].round(4) * 100
        st.dataframe(new)
        boucle_resultats(new)