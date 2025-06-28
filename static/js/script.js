// script.js

document.addEventListener('DOMContentLoaded', function() {
    // Código para el gráfico de confianza si estamos en la página de resultados
    const confidenceValueElement = document.getElementById('confidenceValue');
    if (confidenceValueElement) {
        const confidencePercentage = parseFloat(confidenceValueElement.textContent);

        // Asegurarse de que el valor está entre 0 y 100
        const confidence = Math.max(0, Math.min(100, confidencePercentage));
        const errorMargin = 100 - confidence; // El resto del porcentaje

        const ctx = document.getElementById('resultChart').getContext('2d');
        
        // Define colores para el gráfico
        const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--bs-primary').trim() || '#0d6efd'; // Intenta obtener el color primario de Bootstrap
        const dangerColor = getComputedStyle(document.documentElement).getPropertyValue('--bs-danger').trim() || '#dc3545'; // Color para el error/riesgo
        const successColor = getComputedStyle(document.documentElement).getPropertyValue('--bs-success').trim() || '#198754'; // Color para no riesgo

        // Si el resultado es "Preocupación por salud mental", usa el color de peligro para la confianza
        // Si el resultado es "No preocupación", usa el color de éxito para la confianza
        const resultTextElement = document.querySelector('.result-display h3');
        let confidenceDisplayColor = primaryColor; // Color por defecto
        if (resultTextElement) {
            if (resultTextElement.classList.contains('text-danger')) {
                confidenceDisplayColor = dangerColor; // Si es "Preocupación", usa rojo
            } else if (resultTextElement.classList.contains('text-success')) {
                confidenceDisplayColor = successColor; // Si es "No preocupación", usa verde
            }
        }


        const resultChart = new Chart(ctx, {
            type: 'doughnut', // Gráfico de tipo donut
            data: {
                labels: ['Confianza en el resultado', 'Margen de error'],
                datasets: [{
                    data: [confidence, errorMargin],
                    backgroundColor: [
                        confidenceDisplayColor, // Color para la confianza
                        '#e0e0e0' // Un color gris claro para el margen de error
                    ],
                    borderColor: [
                        '#ffffff', // Borde blanco para separar las secciones
                        '#ffffff'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // Permitir que el gráfico se ajuste al contenedor
                cutout: '70%', // Tamaño del "agujero" central para hacerlo un donut
                plugins: {
                    legend: {
                        position: 'bottom', // Leyenda abajo del gráfico
                        labels: {
                            font: {
                                size: 14 // Tamaño de fuente para las etiquetas de la leyenda
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += context.parsed + '%';
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    // Puedes añadir aquí más lógica JavaScript para otros formularios o interacciones
    // Por ejemplo, para manejar el envío del formulario principal o mostrar mensajes.
    const surveyForm = document.getElementById('surveyForm');
    if (surveyForm) {
        surveyForm.addEventListener('submit', function(event) {
            // Aquí puedes añadir validaciones adicionales o un spinner de carga
            // Por ahora, simplemente permite el envío normal
            console.log('Formulario enviado');
        });
    }
});
