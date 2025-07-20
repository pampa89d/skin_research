# Фаза 2 • Неделя 8 • Пятница
## Проект • Введение в нейронные сети

0. В соответствии с [инструкцией](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/directory_structure.md) создайте git-репозиторий `nn_project` и добавьте туда членов команды. 

1. Создайте страницу, позволяющую классифицировать картинки из датасета [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) моделью ResNet50 (можно использовать любой другой вариант) (подгружаем предобученную часть из `torchvision.models` , заменяем последний слой, замораживаем все параметры кроме классификационного слоя)
2. Создайте страницу, позволяющую классифицировать изображения [образований](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?datasetId=174469&searchQuery=pyt) на коже. Можно сделать свою модель, можно дообучить модель из `torchvision`. 
   
    ```
    !kaggle datasets download -d fanconic/skin-cancer-malignant-vs-benign
    ```

### ‼️ Сервис должен быть развернут на серверах streamlit

### Дополнительные функции

1. Добавьте возможность загрузки изображения с помощью ссылки (текстовое поле, в которое пользователь вставляет ссылку, картинка отображается вместе с результатом классификации). 
2. Добавьте возможность загрузки сразу нескольких изображений. 
3. Добавьте визуализацию времени ответа модели: за сколько секунд модель справилась с задачей определения класса изображения?
4. Создайте отдельную страницу с информацией о:
   * процессе обучения модели: кривые обучения и метрик
   * времени обучения
   * составе датасета (число объектов, распределение по классам)
   * значения метрики f1 и confusion matrix (в виде heatmap)



**Рекомендации**

> 🎥 Короткое [видео](https://youtu.be/MNxhy_jBGs4) о том, как использовать собственные веса при деплое с помощью streamlit

> ❓[Как скачать данные с Kaggle в Google Colaboratory](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/kaggle-colab.md)

<details>
<summary>Ограничение способа</summary>

Для использования `api kaggle` необходимо иметь полностью верифицированный аккаунт (в т.ч. с подтвержденным номером телефона)!  

```
!kaggle datasets download -d paultimothymooney/blood-cells
```
Нужно взять файлы, расположенные в `dataset2-master/dataset2-master/images`

Если такого аккаунта нет, то данные можно скачать вручную. 
</details>

> ❓[Как скачать данные с Google Drive в Google Colaboratory?](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/drive-colab.md)

> ❓[Как сохранить веса и модель в pytorch?](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/save_torchmodel.md)

> [Image Classification Web App using PyTorch and Streamlit](https://github.com/denistanjingyu/Image-Classification-Web-App-using-PyTorch-and-Streamlit/tree/main)
