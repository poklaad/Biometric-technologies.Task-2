%% Задание 1: Детектор лиц на основе метода Template Matching

for i=1:6
    % Чтение изображения и шаблона
    face_image_name = 'face_image' + string(i) + '.jpg';
    for j=1:3
        template_name = 'face_template' + string(j) + '.jpg';
        face_image = imread(face_image_name);
        template = imread(template_name);
        
        % Преобразование в оттенки серого
        face_gray = rgb2gray(face_image);
        template_gray = rgb2gray(template);
        
        % Размер шаблона
        [template_height, template_width] = size(template_gray);
        

        % Параметры пирамиды изображений
        num_levels = 2; % Количество масштабирований/2
        max_level = 4; % Максимальный масштаб
        scale_factor = [1/num_levels:1/num_levels:1 1+max_level/num_levels:max_level/num_levels:max_level];  % Коэффициент уменьшения масштаба на каждом уровне пирамиды
        
        % Инициализация переменных для хранения максимальных результатов
        best_corr = -inf;
        best_bbox = [];

        % Построение пирамиды и поиск по шаблону на каждом уровне
        for level = 1:size(scale_factor, 2)
            % Масштабирование исходного изображения
            scaled_image = imresize(face_gray, scale_factor(level));
            
            % Пропускаем слишком маленькие изображения
            if size(scaled_image, 1) < template_height || size(scaled_image, 2) < template_width
                continue;
            end
            
            % Выполнение Template Matching
            correlation_map = normxcorr2(template_gray, scaled_image);
            
            % Поиск координат максимальной корреляции
            [max_corr, max_idx] = max(abs(correlation_map(:)));
            [y_peak, x_peak] = ind2sub(size(correlation_map), max_idx);
            
            % Вычисление координат привязки относительно оригинала
            y_offset = round(y_peak/scale_factor(level) - template_height/scale_factor(level));
            x_offset = round(x_peak/scale_factor(level) - template_width/scale_factor(level));
            
            % Если текущая корреляция лучше предыдущей, сохраняем результат
            if max_corr > best_corr
                best_corr = max_corr;
                best_bbox = [x_offset, y_offset, round(template_width / scale_factor(level)), round(template_height / scale_factor(level))];
            end
        end
        
        % Отображение результата на оригинальном изображении
        figure, imshow(face_image);
        if ~isempty(best_bbox)
            hold on;
            rectangle('Position', best_bbox, 'EdgeColor', 'r', 'LineWidth', 2);
            title('Template Matching Detection with Scaling');
        else
            title('Template Not Detected');
        end

    end
end

%% Задание 2: Детектор лиц на основе метода Виолы-Джонса

% Загрузка детектора Виолы-Джонса из MATLAB
faceDetector = vision.CascadeObjectDetector();

for j=1:6
    % Чтение изображения
    face_image_name = 'face_image' + string(j) + '.jpg';
    face_image = imread(face_image_name);
    % Обнаружение лиц
    bboxes = step(faceDetector, face_image);
    
    % Отображение результата
    figure, imshow(face_image);
    hold on;
    for i = 1:size(bboxes, 1)
        rectangle('Position', bboxes(i, :), 'EdgeColor', 'g', 'LineWidth', 2);
    end
    title('Viola-Jones Face Detection');
end

%% Задание 3: Определение линии симметрии лица

for j=1:6
    % Чтение изображения
    face_image_name = 'face_image' + string(j) + 'cropped.jpg';
    face_image = imread(face_image_name);
    face_gray = rgb2gray(face_image); % Преобразование в оттенки серого
    
    % Размер изображения
    [M, N] = size(face_gray);
    
    % Границы центральной части лица
    Y1 = 1;
    Y2 = M;
    X1 = 1;
    X2 = N;
    
    % Параметры для полос L и R
    w = 150; % Ширина полос для центральной оси
    d_min = inf;
    Xc = X1; % Координата центральной линии симметрии
    
    % Поиск центральной линии симметрии
    for xt = X1 + w : X2 - w
        L_strip = face_gray(Y1:Y2, xt-w:xt-1);
        R_strip = face_gray(Y1:Y2, xt+1:xt+w);
        R_strip_flipped = flip(R_strip, 2); % Отражение полосы R
        
        % Расчет расстояния между полосами L и R
        d_xt = sum(sum(abs(L_strip - R_strip_flipped)));
        
        % Поиск минимального расстояния
        if d_xt < d_min
            d_min = d_xt;
            Xc = xt;
        end
    end
    
    % Отображение центральной линии симметрии
    figure, imshow(face_image);
    hold on;
    line([Xc, Xc], [Y1, Y2], 'Color', 'r', 'LineWidth', 2);
    title('Central Symmetry Line');
    
    % Поиск локальных линий симметрии для областей глаз
    w_eye = 75; % Уменьшение ширины полос для локальных линий
    X_left_eye = X1;
    X_right_eye = X2;
    
    % Локальная линия для левого глаза
    d_min_left = inf;
    for xt = X1 + w_eye : Xc - w_eye
        L_strip = face_gray(Y1:Y2, xt-w_eye:xt-1);
        R_strip = face_gray(Y1:Y2, xt+1:xt+w_eye);
        R_strip_flipped = flip(R_strip, 2);
        
        d_xt = sum(sum(abs(L_strip - R_strip_flipped)));
        if d_xt < d_min_left
            d_min_left = d_xt;
            X_left_eye = xt;
        end
    end
    
    % Локальная линия для правого глаза
    d_min_right = inf;
    for xt = Xc + w_eye : X2 - w_eye
        L_strip = face_gray(Y1:Y2, xt-w_eye:xt-1);
        R_strip = face_gray(Y1:Y2, xt+1:xt+w_eye);
        R_strip_flipped = flip(R_strip, 2);
        
        d_xt = sum(sum(abs(L_strip - R_strip_flipped)));
        if d_xt < d_min_right
            d_min_right = d_xt;
            X_right_eye = xt;
        end
    end
    
    % Отображение всех линий симметрии
    line([X_left_eye, X_left_eye], [Y1, Y2], 'Color', 'g', 'LineWidth', 2);
    line([X_right_eye, X_right_eye], [Y1, Y2], 'Color', 'b', 'LineWidth', 2);
    legend('Central Symmetry Line', 'Left Eye Symmetry Line', 'Right Eye Symmetry Line');

end
