
# ------------------------------------------
# Clasificación con Árbol de Decisión - Dataset Iris
# ------------------------------------------

# Cargar librerías necesarias
library(datasets)
library(tree)

# Cargar los datos
data(iris)

# Seleccionar solo las variables requeridas
iris_sub <- iris[, c("Species", "Sepal.Width", "Petal.Width")]

# Ver estructura
str(iris_sub)

# Establecer semilla para reproducibilidad
set.seed(123)

# Dividir los datos en entrenamiento (70%) y prueba (30%)
alpha <- 0.7
n <- nrow(iris_sub)
train_index <- sample(1:n, size = alpha * n)

train_data <- iris_sub[train_index, ]
test_data <- iris_sub[-train_index, ]

# ------------------------------
# Crear el árbol de clasificación
# ------------------------------
arbol <- tree(Species ~ Sepal.Width + Petal.Width, data = train_data)

# Mostrar resumen del árbol
cat("Resumen del árbol:\n")
print(summary(arbol))

# ------------------------------
# Predicciones de probabilidad
# ------------------------------
pred_prob <- predict(arbol, test_data, type = "prob")

cat("\nProbabilidades de clasificación (primeras observaciones):\n")
print(head(pred_prob))

# ------------------------------
# Predicciones de clases
# ------------------------------
pred_class <- predict(arbol, test_data, type = "class")

cat("\nPredicciones observadas vs predichas:\n")
print(head(data.frame(Observado = test_data$Species, Predicho = pred_class)))

# ------------------------------
# Visualizar el árbol
# ------------------------------
plot(arbol)
text(arbol, pretty = 0)

# ------------------------------
# Poda del árbol a 4 hojas
# ------------------------------
cv_arbol <- cv.tree(arbol, FUN = prune.tree)
cat("\nResultados de validación cruzada:\n")
print(cv_arbol)

# Podar a 4 hojas
arbol_podado <- prune.tree(arbol, best = 4)

# Visualizar árbol podado
plot(arbol_podado)
text(arbol_podado, pretty = 0)

# Predicciones con el árbol podado
pred_podado <- predict(arbol_podado, test_data, type = "class")

cat("\nPredicciones con árbol podado:\n")
print(head(data.frame(Observado = test_data$Species, Predicho = pred_podado)))

# ------------------------------
# Evaluación del desempeño
# ------------------------------
cat("\nPrecisión antes de la poda: ")
print(mean(pred_class == test_data$Species))

cat("Precisión después de la poda: ")
print(mean(pred_podado == test_data$Species))
