import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_regression(x, phi0, phi1):
    """
    Логистическая регрессия: P(y=1|x) = sigmoid(φ₀ + φ₁x)
    Возвращает вероятность класса 1 для каждого x
    """
    return sigmoid(phi0 + phi1 * x)

def plot_model(x, y_data, x_test=None):
    """
    Визуализация логистической регрессии: P(y=1|x) = sigmoid(φ₀ + φ₁x)
    
    ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ:
    ========================
    Логистическая регрессия используется для бинарной классификации (класс 0 или класс 1).
    
    Формула: P(y=1|x) = sigmoid(φ₀ + φ₁x) = 1 / (1 + exp(-(φ₀ + φ₁x)))
    
    Параметры модели:
    • φ₀ (phi0, intercept/bias) - смещение
      - Сдвигает логистическую кривую влево (φ₀ > 0) или вправо (φ₀ < 0)
      - Определяет точку, где вероятность класса 1 = 0.5 (решающая граница)
      - При x = -φ₀/φ₁ вероятность равна 0.5
    
    • φ₁ (phi1, slope/weight) - наклон/вес признака
      - Определяет крутизну логистической кривой
      - φ₁ > 0: вероятность класса 1 растет с увеличением x
      - φ₁ < 0: вероятность класса 1 убывает с увеличением x
      - |φ₁| больше → переход более резкий (более уверенное разделение классов)
      - |φ₁| меньше → переход более плавный (менее уверенное разделение)
    
    РЕШАЮЩАЯ ГРАНИЦА:
    ----------------
    При P(y=1|x) = 0.5 происходит разделение классов:
    - P(y=1|x) > 0.5 → предсказываем класс 1
    - P(y=1|x) < 0.5 → предсказываем класс 0
    - Точка x = -φ₀/φ₁ является решающей границей
    """
    if x_test is None:
        x_test = np.linspace(-6, 6, 200)  # Уменьшаем количество точек для быстродействия
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Логистическая регрессия: P(y=1|x) = sigmoid(φ₀ + φ₁x)', fontsize=16, fontweight='bold')
    
    # График 1: Обзор различных логистических кривых
    # Показывает, как разные комбинации параметров создают разные формы S-кривых
    ax1 = axes[0, 0]
    # Показываем данные для классификации
    if y_data is not None:
        # Разделяем точки классов 0 и 1 для правильной легенды
        mask_class0 = y_data == 0
        mask_class1 = y_data == 1
        ax1.scatter(x[mask_class0], y_data[mask_class0], s=50, c='red', zorder=5, 
                   label='Класс 0', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[mask_class1], y_data[mask_class1], s=50, c='blue', zorder=5, 
                   label='Класс 1', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Уменьшаем количество кривых для первого графика (только репрезентативные примеры)
    example_combinations = [
        (-2, 1.0), (-1, 1.0), (0, 1.0), (1, 1.0), (2, 1.0),  # Разные phi0, одинаковый phi1
        (0, 0.5), (0, 1.0), (0, 2.0),  # Разные phi1, одинаковый phi0=0
        (-1, -1.0), (1, -1.0),  # Обратные наклоны
    ]
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(example_combinations)))
    
    for idx, (phi0, phi1) in enumerate(example_combinations):
        prob = logistic_regression(x_test, phi0, phi1)
        ax1.plot(x_test, prob, color=colors[idx], alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('x (признак)', fontsize=12)
    ax1.set_ylabel('P(y=1|x) (вероятность класса 1)', fontsize=12)
    ax1.set_title('Различные логистические кривые', fontsize=12)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Решающая граница (0.5)')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax1.legend(loc='best', fontsize=8)
    
    # График 2: Влияние φ₀ (intercept/bias) - сдвиг логистической кривой
    # При фиксированном φ₁ показываем, как φ₀ сдвигает кривую влево/вправо
    # Решающая граница (вероятность = 0.5) перемещается: x_boundary = -φ₀/φ₁
    ax2 = axes[0, 1]
    if y_data is not None:
        mask_class0 = y_data == 0
        mask_class1 = y_data == 1
        ax2.scatter(x[mask_class0], y_data[mask_class0], s=50, c='red', zorder=5, 
                   label='Класс 0', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.scatter(x[mask_class1], y_data[mask_class1], s=50, c='blue', zorder=5, 
                   label='Класс 1', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    phi1_fixed = 1.0  # Фиксируем наклон
    phi0_range = np.linspace(-3, 3, 5)  # Уменьшаем количество до 5 кривых
    colors_phi0 = plt.get_cmap('plasma')(np.linspace(0, 1, len(phi0_range)))
    
    for i, phi0 in enumerate(phi0_range):
        prob = logistic_regression(x_test, phi0, phi1_fixed)
        x_boundary = -phi0 / phi1_fixed if phi1_fixed != 0 else 0
        ax2.plot(x_test, prob, color=colors_phi0[i], linewidth=2.5, 
                label=f'φ₀={phi0:.1f} → x={x_boundary:.1f}', alpha=0.8)
        # Отмечаем решающую границу
        if -6 <= x_boundary <= 6:
            ax2.plot(x_boundary, 0.5, 'o', color=colors_phi0[i], markersize=7, 
                    markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_xlabel('x (признак)', fontsize=12)
    ax2.set_ylabel('P(y=1|x)', fontsize=12)
    ax2.set_title(f'Влияние φ₀ (intercept): сдвиг кривой при φ₁={phi1_fixed:.1f}', fontsize=12)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax2.legend(fontsize=7, loc='best', ncol=1, framealpha=0.9)
    
    # График 3: Влияние φ₁ (slope/weight) - крутизна логистической кривой
    # При фиксированном φ₀ показываем, как φ₁ изменяет крутизну перехода
    # Больший |φ₁| → более резкий переход (лучше разделение классов)
    # Меньший |φ₁| → более плавный переход (менее уверенное разделение)
    ax3 = axes[1, 0]
    if y_data is not None:
        mask_class0 = y_data == 0
        mask_class1 = y_data == 1
        ax3.scatter(x[mask_class0], y_data[mask_class0], s=50, c='red', zorder=5, 
                   label='Класс 0', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax3.scatter(x[mask_class1], y_data[mask_class1], s=50, c='blue', zorder=5, 
                   label='Класс 1', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    phi0_fixed = 0.0  # Фиксируем intercept (граница при x=0)
    phi1_range = np.array([-2, -1, -0.5, 0.5, 1, 2])  # Уменьшаем до 6 значений
    colors_phi1 = plt.get_cmap('coolwarm')(np.linspace(0, 1, len(phi1_range)))
    
    for i, phi1 in enumerate(phi1_range):
        prob = logistic_regression(x_test, phi0_fixed, phi1)
        x_boundary = -phi0_fixed / phi1 if phi1 != 0 else 0
        label = f'φ₁={phi1:.1f}'
        if phi1 > 0:
            label += ' (↑)'
        elif phi1 < 0:
            label += ' (↓)'
        ax3.plot(x_test, prob, color=colors_phi1[i], linewidth=2.5, 
                label=label, alpha=0.8)
        # Отмечаем решающую границу
        if abs(phi1) > 0.1 and -6 <= x_boundary <= 6:
            ax3.plot(x_boundary, 0.5, 'o', color=colors_phi1[i], markersize=7, 
                    markeredgecolor='black', markeredgewidth=0.5)
    
    ax3.set_xlabel('x (признак)', fontsize=12)
    ax3.set_ylabel('P(y=1|x)', fontsize=12)
    ax3.set_title(f'Влияние φ₁ (slope): крутизна перехода при φ₀={phi0_fixed:.1f}', fontsize=12)
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax3.legend(fontsize=7, loc='best', ncol=2, framealpha=0.9)
    
    # График 4: Репрезентативные комбинации - демонстрация совместного влияния
    # Показывает ключевые комбинации (φ₀, φ₁), демонстрируя разные сценарии классификации
    ax4 = axes[1, 1]
    if y_data is not None:
        mask_class0 = y_data == 0
        mask_class1 = y_data == 1
        ax4.scatter(x[mask_class0], y_data[mask_class0], s=50, c='red', zorder=5, 
                   label='Класс 0', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.scatter(x[mask_class1], y_data[mask_class1], s=50, c='blue', zorder=5, 
                   label='Класс 1', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    selected_combinations = [
        # Разные положения границы и крутизны: (phi0, phi1)
        (-2, 0.5),
        (0, 1.0),
        (2, 2.0),
        (-1, -1.0),
        (1, -0.5),
    ]
    
    colors_comb = plt.get_cmap('Set2')(np.linspace(0, 1, len(selected_combinations)))
    for idx, (phi0, phi1) in enumerate(selected_combinations):
        prob = logistic_regression(x_test, phi0, phi1)
        x_boundary = -phi0 / phi1 if phi1 != 0 else None
        ax4.plot(x_test, prob, linewidth=2.5, color=colors_comb[idx], 
                label=f'φ₀={phi0}, φ₁={phi1:.1f}', alpha=0.8)
        if x_boundary is not None and -6 <= x_boundary <= 6:
            ax4.plot(x_boundary, 0.5, 'o', color=colors_comb[idx], markersize=8, 
                    markeredgecolor='black', markeredgewidth=1, zorder=10)
            # Упрощенная аннотация для границы
            ax4.text(x_boundary, 0.6, f'{x_boundary:.1f}', fontsize=7, 
                    ha='center', color=colors_comb[idx], weight='bold')
    
    ax4.set_xlabel('x (признак)', fontsize=12)
    ax4.set_ylabel('P(y=1|x)', fontsize=12)
    ax4.set_title('Различные сценарии классификации', fontsize=12)
    ax4.set_xlim(-6, 6)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1, 
               label='Решающая граница')
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax4.legend(fontsize=7, loc='best', ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

def generate_classification_data(n_samples=50, noise=0.1):
    """
    Генерирует синтетические данные для бинарной классификации
    """
    np.random.seed(42)
    x = np.linspace(-5, 5, n_samples)
    # Создаем данные с разделением: x < 0 → класс 0, x > 0 → класс 1 (с некоторым шумом)
    y_true = (x > 0).astype(int)
    # Добавляем шум: некоторые точки могут быть неправильно классифицированы
    y = y_true.copy()
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = 1 - y[noise_mask]  # Инвертируем некоторые метки
    return x, y

def generate_data_for_loss():
    """
    Генерирует данные для визуализации loss:
    - 10 точек из N(-1, 1) с меткой y=0
    - 10 точек из N(1, 1) с меткой y=1
    """
    np.random.seed(42)
    # Класс 0: mean=-1, std=1
    x_class0 = np.random.normal(loc=-1.0, scale=1.0, size=10)
    y_class0 = np.zeros(10)
    
    # Класс 1: mean=1, std=1
    x_class1 = np.random.normal(loc=1.0, scale=1.0, size=10)
    y_class1 = np.ones(10)
    
    # Объединяем
    x = np.concatenate([x_class0, x_class1])
    y = np.concatenate([y_class0, y_class1])
    
    return x, y

def binary_cross_entropy_loss(x, y, phi0, phi1, eps=1e-15):
    """
    Вычисляет Binary Cross Entropy loss для логистической регрессии
    
    L = -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
    где p_i = sigmoid(phi0 + phi1*x_i)
    
    Args:
        x: массив признаков (N,)
        y: массив меток (N,), значения 0 или 1
        phi0: параметр intercept
        phi1: параметр slope
        eps: маленькое значение для численной стабильности
    
    Returns:
        значение функции потерь
    """
    # Вычисляем вероятности
    p = logistic_regression(x, phi0, phi1)
    
    # Обеспечиваем численную стабильность
    p = np.clip(p, eps, 1 - eps)
    
    # Binary Cross Entropy
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    return loss

def plot_loss_heatmap(x_data=None, y_data=None, phi0_range=None, phi1_range=None, n_points=40):
    """
    Строит heatmap функции потерь в зависимости от параметров φ₀ и φ₁
    
    Args:
        x_data: массив признаков (если None, генерируются новые данные)
        y_data: массив меток (если None, генерируются новые данные)
        phi0_range: диапазон для φ₀ (min, max) или None для автоопределения
        phi1_range: диапазон для φ₁ (min, max) или None для автоопределения
        n_points: количество точек на каждой оси для сетки
    """
    # Генерируем данные, если не предоставлены
    if x_data is None or y_data is None:
        x_data, y_data = generate_data_for_loss()
    
    # Определяем диапазоны параметров
    if phi0_range is None:
        phi0_min, phi0_max = -5, 5
    else:
        phi0_min, phi0_max = phi0_range
    
    if phi1_range is None:
        phi1_min, phi1_max = -5, 5
    else:
        phi1_min, phi1_max = phi1_range
    
    # Создаем сетку параметров
    phi0_vals = np.linspace(phi0_min, phi0_max, n_points)
    phi1_vals = np.linspace(phi1_min, phi1_max, n_points)
    PHI0, PHI1 = np.meshgrid(phi0_vals, phi1_vals)
    
    # Вычисляем loss для каждой комбинации параметров (векторизованная версия)
    print("Вычисление функции потерь...")
    LOSS = np.zeros_like(PHI0)
    
    # Векторизуем вычисления для ускорения
    x_data_expanded = x_data.reshape(-1, 1, 1)  # (N, 1, 1)
    y_data_expanded = y_data.reshape(-1, 1, 1)  # (N, 1, 1)
    phi0_expanded = PHI0.reshape(1, n_points, n_points)  # (1, H, W)
    phi1_expanded = PHI1.reshape(1, n_points, n_points)  # (1, H, W)
    
    # Вычисляем вероятности для всех комбинаций одновременно
    logits = phi0_expanded + phi1_expanded * x_data_expanded  # (N, H, W)
    p = sigmoid(logits)
    
    # Обеспечиваем численную стабильность
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    
    # Вычисляем loss для каждого параметра
    # LOSS имеет форму (H, W), нам нужно усреднить по N (первая ось)
    log_loss = -(y_data_expanded * np.log(p) + (1 - y_data_expanded) * np.log(1 - p))
    LOSS = np.mean(log_loss, axis=0)
    
    # Строим heatmap
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Используем contourf для плавного heatmap
    levels = 50
    contour = ax.contourf(PHI0, PHI1, LOSS, levels=levels, cmap='viridis_r', alpha=0.9)
    contour_lines = ax.contour(PHI0, PHI1, LOSS, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Цветовая шкала
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Binary Cross Entropy Loss', fontsize=12, rotation=270, labelpad=20)
    
    # Находим минимум
    min_idx = np.unravel_index(np.argmin(LOSS), LOSS.shape)
    phi0_opt = PHI0[min_idx]
    phi1_opt = PHI1[min_idx]
    loss_min = LOSS[min_idx]
    
    # Отмечаем точку минимума
    ax.plot(phi0_opt, phi1_opt, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label=f'Минимум: φ₀={phi0_opt:.2f}, φ₁={phi1_opt:.2f}, Loss={loss_min:.3f}', zorder=10)
    
    # Добавляем информацию о данных в текстовом блоке
    data_info = (f'Данные:\n'
                 f'• Класс 0: 10 точек из N(μ=-1, σ=1)\n'
                 f'  x ∈ [{x_data[y_data==0].min():.2f}, {x_data[y_data==0].max():.2f}]\n'
                 f'• Класс 1: 10 точек из N(μ=1, σ=1)\n'
                 f'  x ∈ [{x_data[y_data==1].min():.2f}, {x_data[y_data==1].max():.2f}]')
    
    ax.text(0.02, 0.98, data_info, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Показываем данные на графике
    ax.set_xlabel('φ₀ (intercept)', fontsize=12, fontweight='bold')
    ax.set_ylabel('φ₁ (slope)', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap функции потерь (Binary Cross Entropy)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    return phi0_opt, phi1_opt, loss_min, LOSS

def main():
    # Генерируем данные для классификации
    x_data, y_data = generate_classification_data(n_samples=50, noise=0.15)
    
    # Создаем сетку для плавного отображения кривых (уменьшено для быстродействия)
    x_smooth = np.linspace(-6, 6, 200)
    
    # plot_model(x_data, y_data, x_test=x_smooth)
    
    # Пример использования функции для heatmap
    plot_loss_heatmap()

if __name__ == "__main__":
    main()
