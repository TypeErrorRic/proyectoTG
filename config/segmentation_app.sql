-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 08-12-2025 a las 20:20:41
-- Versión del servidor: 10.4.32-MariaDB
-- Versión de PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `segmentation_app`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `captures`
--

CREATE TABLE `captures` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `configuration_id` int(10) UNSIGNED DEFAULT NULL,
  `filename` varchar(255) NOT NULL,
  `image_data` longblob DEFAULT NULL,
  `file_size_bytes` int(11) DEFAULT NULL,
  `image_width` int(11) DEFAULT NULL,
  `image_height` int(11) DEFAULT NULL,
  `mode` enum('camera','dataset') DEFAULT 'camera',
  `dataset_index` int(11) DEFAULT NULL,
  `ransac_time_ms` decimal(8,2) DEFAULT NULL,
  `fps` decimal(6,2) DEFAULT NULL,
  `num_ground_pixels` longblob DEFAULT NULL,
  `num_wall_pixels` int(11) DEFAULT NULL,
  `num_door_pixels` int(11) DEFAULT NULL,
  `tags` varchar(255) DEFAULT NULL,
  `notes` text DEFAULT NULL,
  `captured_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `is_favorite` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `configurations`
--

CREATE TABLE `configurations` (
  `id` int(10) UNSIGNED NOT NULL,
  `user_id` int(10) UNSIGNED NOT NULL,
  `config_name` varchar(100) NOT NULL,
  `description` text DEFAULT NULL,
  `subsample_stride` int(11) DEFAULT 1,
  `dist_thresh` decimal(6,4) DEFAULT 0.0300,
  `max_iters` int(11) DEFAULT 400,
  `min_inliers` int(11) DEFAULT 400,
  `max_angle_deg` decimal(5,2) DEFAULT 60.00,
  `score_subset` int(11) DEFAULT 4096,
  `time_budget_ms` decimal(7,2) DEFAULT 120.00,
  `early_stop_ratio` decimal(4,3) DEFAULT 0.920,
  `batch_size` int(11) DEFAULT 128,
  `low_height_pct` decimal(5,2) DEFAULT 25.00,
  `roi_bottom_fraction` decimal(4,3) DEFAULT 0.340,
  `roi_expand_step` decimal(4,3) DEFAULT 0.200,
  `max_agg_points` int(11) DEFAULT 150000,
  `refine_full_res` tinyint(1) DEFAULT 1,
  `refine_max_points` int(11) DEFAULT 200000,
  `refine_dist_mult` decimal(4,2) DEFAULT 1.60,
  `second_pass_mask` tinyint(1) DEFAULT 1,
  `is_default` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Volcado de datos para la tabla `configurations`
--

INSERT INTO `configurations` (`id`, `user_id`, `config_name`, `description`, `subsample_stride`, `dist_thresh`, `max_iters`, `min_inliers`, `max_angle_deg`, `score_subset`, `time_budget_ms`, `early_stop_ratio`, `batch_size`, `low_height_pct`, `roi_bottom_fraction`, `roi_expand_step`, `max_agg_points`, `refine_full_res`, `refine_max_points`, `refine_dist_mult`, `second_pass_mask`, `is_default`, `created_at`, `updated_at`) VALUES
(1, 1, 'Default Configuration', 'Factory default parameters', 1, 0.0300, 400, 400, 60.00, 4096, 120.00, 0.920, 128, 25.00, 0.340, 0.200, 150000, 1, 200000, 1.60, 1, 1, '2025-12-08 19:19:23', '2025-12-08 19:19:23');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `users`
--

CREATE TABLE `users` (
  `id` int(10) UNSIGNED NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `full_name` varchar(100) DEFAULT NULL,
  `role` enum('admin','operator','viewer') DEFAULT 'operator',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `last_login` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Volcado de datos para la tabla `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `password_hash`, `full_name`, `role`, `created_at`, `last_login`) VALUES
(1, 'admin', 'admin@segmentation.local', '0192023a7bbd73250516f069df18b500', 'Administrator', 'admin', '2025-12-08 19:19:23', NULL);

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `captures`
--
ALTER TABLE `captures`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `configuration_id` (`configuration_id`);

--
-- Indices de la tabla `configurations`
--
ALTER TABLE `configurations`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`,`config_name`);

--
-- Indices de la tabla `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `captures`
--
ALTER TABLE `captures`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT de la tabla `configurations`
--
ALTER TABLE `configurations`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT de la tabla `users`
--
ALTER TABLE `users`
  MODIFY `id` int(10) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- Restricciones para tablas volcadas
--

--
-- Filtros para la tabla `captures`
--
ALTER TABLE `captures`
  ADD CONSTRAINT `captures_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `captures_ibfk_2` FOREIGN KEY (`configuration_id`) REFERENCES `configurations` (`id`) ON DELETE SET NULL;

--
-- Filtros para la tabla `configurations`
--
ALTER TABLE `configurations`
  ADD CONSTRAINT `configurations_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
