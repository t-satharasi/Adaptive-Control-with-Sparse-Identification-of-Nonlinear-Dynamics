classdef ICLAffine < handle
    properties
        P
        fun_Y
        fun_g
        fun_f0
        fun_gplus
        fun_u
        fun_xd
        fun_xd_dot
        sol
    end

    properties (SetAccess = private)
        tau
        lambda
        script_Y
        script_FGu
        script_X
    end

    methods
        function self = ICLAffine(params)
            if nargin < 1
                params = struct();
            end

            self.P = self.defaultParameters();
            self.P = self.applyOverrides(self.P, params);

            self.configureModel();
            self.validateParameters();

            self.sol = struct('t', [], 'y', []);
            self.tau = 0;
            self.script_Y = zeros(self.P.n * self.P.N, self.P.p);
            self.script_FGu = zeros(self.P.n * self.P.N, 1);
            self.script_X = zeros(self.P.n * self.P.N, 1);
            self.lambda = 0;
        end

        function simulate(self)
            n = self.P.n;
            p = self.P.p;
            N = self.P.N; %#ok<NASGU>

            self.tau = 0;
            self.script_Y = zeros(n * self.P.N, p);
            self.script_FGu = zeros(n * self.P.N, 1);
            self.script_X = zeros(n * self.P.N, 1);
            self.lambda = 0;

            options = ddeset( ...
                'Events', @(t,y,Z) self.ddefunEvents(t,y,Z), ...
                'RelTol', self.P.rel_tol, ...
                'AbsTol', self.P.abs_tol);

            t0 = 0;
            initial_state = [ ...
                self.P.x_0; ...
                self.P.theta_hat_0; ...
                zeros(n * p, 1); ...
                zeros(n, 1)];

            self.sol.t = [];
            self.sol.y = [];

            while t0 < self.P.t_f
                solution = dde23( ...
                    @(t,y,Z) self.ddefunContinuous(t,y,Z), ...
                    self.P.T, ...
                    @(t) self.ddefunHistory(t, initial_state), ...
                    [t0, self.P.t_f], ...
                    options);

                initial_state = solution.y(:, end);
                t0 = solution.x(end);

                if numel(solution.x) > 1
                    self.sol.t = [self.sol.t, solution.x(1:end-1)]; %#ok<AGROW>
                    self.sol.y = [self.sol.y, solution.y(:,1:end-1)]; %#ok<AGROW>
                else
                    break;
                end

                if t0 >= self.P.t_f
                    self.sol.t = [self.sol.t, solution.x(end)]; %#ok<AGROW>
                    self.sol.y = [self.sol.y, solution.y(:,end)]; %#ok<AGROW>
                    break;
                end
            end
        end

        function metrics = plotsimulations(self, varargin)
            self.assertSimulationAvailable();

            opts = self.parsePlotOptions(varargin{:});

            t = self.sol.t(:);
            y = self.sol.y;
            n = self.P.n;
            p = self.P.p;
            idx = self.stateIndices();

            x = y(idx.x, :);
            xd = self.referenceTrajectory(t.');
            e = x - xd;

            theta_hat = y(idx.theta_hat, :);
            theta_true = self.P.theta;
            theta_hat_final = theta_hat(:, end);
            theta_error = theta_hat - theta_true;

            [support_metrics, threshold_table] = ...
                self.computeSupportMetrics(theta_true, theta_hat_final);

            tracking_error_norm = vecnorm(e, 2, 1);
            final_tracking_error_norm = norm(e(:, end));

            if numel(t) > 1 && (t(end) > t(1))
                rms_tracking_error = sqrt(trapz(t, sum(e.^2, 1).') / (t(end) - t(1)));
            else
                rms_tracking_error = norm(e(:, end));
            end

            final_parameter_error_norm = norm(theta_error(:, end));
            coeff_table = self.coefficientTable(theta_true, theta_hat_final);

            metrics = struct();
            metrics.lambda = self.P.sparsity_lambda;
            metrics.sparsity_tol = self.P.sparsity_tol;
            metrics.true_nonzeros = support_metrics.true_nonzeros;
            metrics.estimated_nonzeros = support_metrics.estimated_nonzeros;
            metrics.true_zeros = support_metrics.true_zeros;
            metrics.estimated_zeros = support_metrics.estimated_zeros;
            metrics.tp = support_metrics.tp;
            metrics.fp = support_metrics.fp;
            metrics.fn = support_metrics.fn;
            metrics.tn = support_metrics.tn;
            metrics.final_tracking_error_norm = final_tracking_error_norm;
            metrics.rms_tracking_error = rms_tracking_error;
            metrics.final_parameter_error_norm = final_parameter_error_norm;
            metrics.theta_hat_final = theta_hat_final;
            metrics.theta_true = theta_true;
            metrics.threshold_table = threshold_table;
            metrics.coefficient_table = coeff_table;

            lambda_label = self.lambdaLabel(self.P.sparsity_lambda);

            if opts.Visible
                fig_visible = 'on';
            else
                fig_visible = 'off';
            end

            figure_handles_main = gobjects(4,1);
            figure_names_main = { ...
                sprintf('paper_state_tracking_%s', lambda_label), ...
                sprintf('paper_tracking_error_norm_%s', lambda_label), ...
                sprintf('paper_coefficients_%s', lambda_label), ...
                sprintf('paper_identification_vs_truth_%s', lambda_label)};

            figure_handles_appendix = gobjects(2,1);
            figure_names_appendix = { ...
                sprintf('appendix_tracking_error_components_%s', lambda_label), ...
                sprintf('appendix_parameter_error_%s', lambda_label)};

            labels = self.libraryLabels();
            block1 = 1:(p/2);
            block2 = (p/2+1):p;

            %% Main Figure 1
            figure_handles_main(1) = figure('Visible', fig_visible);
            clf;

            subplot(2,1,1);
            plot(t, x(1,:), 'LineWidth', 1.5, 'DisplayName', '$x_1$');
            hold on;
            plot(t, xd(1,:), '--', 'LineWidth', 1.5, 'DisplayName', '$x_{d,1}$');
            hold off;
            xlabel('Time');
            ylabel('State');
            title(sprintf('State Tracking, $x_1$ ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            legend('Interpreter', 'latex', 'Location', 'best');
            grid minor;

            subplot(2,1,2);
            plot(t, x(2,:), 'LineWidth', 1.5, 'DisplayName', '$x_2$');
            hold on;
            plot(t, xd(2,:), '--', 'LineWidth', 1.5, 'DisplayName', '$x_{d,2}$');
            hold off;
            xlabel('Time');
            ylabel('State');
            title(sprintf('State Tracking, $x_2$ ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            legend('Interpreter', 'latex', 'Location', 'best');
            grid minor;

            %% Main Figure 2
            figure_handles_main(2) = figure('Visible', fig_visible);
            clf;
            plot(t, tracking_error_norm, 'LineWidth', 1.6);
            xlabel('Time');
            ylabel('$\|e(t)\|$', 'Interpreter', 'latex');
            title(sprintf('Tracking Error Norm ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            grid minor;

            %% Main Figure 3
            figure_handles_main(3) = figure('Visible', fig_visible);
            clf;

            subplot(2,1,1);
            stem(1:numel(block1), abs(theta_true(block1)), 'filled', ...
                'LineWidth', 1.1, 'DisplayName', '$|\theta|$');
            hold on;
            stem(1:numel(block1), abs(theta_hat_final(block1)), '--', ...
                'LineWidth', 1.1, 'DisplayName', '$|\hat{\theta}(t_f)|$');
            for k = 1:numel(self.P.support_thresholds)
                yline(self.P.support_thresholds(k), ':', 'HandleVisibility', 'off');
            end
            hold off;
            set(gca, 'YScale', 'log');
            xticks(1:numel(block1));
            xticklabels(labels(block1));
            ax = gca;
            ax.TickLabelInterpreter = 'latex';
            xtickangle(30);
            ylabel('Magnitude');
            title('Recovered Coefficients for $\dot{x}_1$', 'Interpreter', 'latex');
            legend('Interpreter', 'latex', 'Location', 'best');
            grid minor;

            subplot(2,1,2);
            stem(1:numel(block2), abs(theta_true(block2)), 'filled', ...
                'LineWidth', 1.1, 'DisplayName', '$|\theta|$');
            hold on;
            stem(1:numel(block2), abs(theta_hat_final(block2)), '--', ...
                'LineWidth', 1.1, 'DisplayName', '$|\hat{\theta}(t_f)|$');
            for k = 1:numel(self.P.support_thresholds)
                yline(self.P.support_thresholds(k), ':', 'HandleVisibility', 'off');
            end
            hold off;
            set(gca, 'YScale', 'log');
            xticks(1:numel(block2));
            xticklabels(labels(block2));
            ax = gca;
            ax.TickLabelInterpreter = 'latex';
            xtickangle(30);
            xlabel('Library term');
            ylabel('Magnitude');
            title('Recovered Coefficients for $\dot{x}_2$', 'Interpreter', 'latex');
            legend('Interpreter', 'latex', 'Location', 'best');
            grid minor;

            %% Main Figure 4
            figure_handles_main(4) = figure('Visible', fig_visible);
            clf;
            stem(1:p, theta_true, 'filled', 'LineWidth', 1.1, ...
                'DisplayName', 'True $\theta$');
            hold on;
            stem(1:p, theta_hat_final, '--', 'LineWidth', 1.1, ...
                'DisplayName', 'Estimated $\hat{\theta}(t_f)$');
            yline(self.P.sparsity_tol, ':', 'HandleVisibility', 'off');
            yline(-self.P.sparsity_tol, ':', 'HandleVisibility', 'off');
            hold off;
            xticks(1:p);
            xticklabels(labels);
            ax = gca;
            ax.TickLabelInterpreter = 'latex';
            xtickangle(45);
            xlabel('Library term');
            ylabel('Coefficient value');
            title(sprintf('Final Identified Coefficients ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            legend('Interpreter', 'latex', 'Location', 'bestoutside');
            grid minor;

            %% Appendix Figure 1
            figure_handles_appendix(1) = figure('Visible', fig_visible);
            clf;
            plot(t, e', 'LineWidth', 1.2);
            xlabel('Time');
            ylabel('$e_i(t)$', 'Interpreter', 'latex');
            title(sprintf('Tracking Error Components ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            legend(compose('$e_{%d}$', 1:n), ...
                'Interpreter', 'latex', 'Location', 'bestoutside');
            grid minor;

            %% Appendix Figure 2
            figure_handles_appendix(2) = figure('Visible', fig_visible);
            clf;
            plot(t, theta_error', 'LineWidth', 1.0);
            xlabel('Time');
            ylabel('$\tilde{\theta}_i(t)$', 'Interpreter', 'latex');
            title(sprintf('Parameter Estimation Error ($\\lambda = %g$)', self.P.sparsity_lambda), ...
                'Interpreter', 'latex');
            grid minor;

            if opts.SaveFigures
                if strlength(string(opts.FigureDir)) == 0
                    error('ICLAffine:InvalidInput', ...
                        'FigureDir must be provided when SaveFigures is true.');
                end

                self.ensureDirectory(opts.FigureDir);
                self.saveFigures(figure_handles_main, figure_names_main, opts.FigureDir);

                if opts.SaveAppendixFigures
                    self.saveFigures(figure_handles_appendix, figure_names_appendix, opts.FigureDir);
                end
            end

            fprintf('\n===== ICLAffine Summary =====\n');
            fprintf('lambda                          : %.6g\n', self.P.sparsity_lambda);
            fprintf('sparsity threshold              : %.2e\n', self.P.sparsity_tol);
            fprintf('true nonzeros                   : %d\n', support_metrics.true_nonzeros);
            fprintf('estimated nonzeros              : %d\n', support_metrics.estimated_nonzeros);
            fprintf('TP / FP / FN / TN               : %d / %d / %d / %d\n', ...
                support_metrics.tp, support_metrics.fp, ...
                support_metrics.fn, support_metrics.tn);
            fprintf('final ||e(tf)||                 : %.6e\n', final_tracking_error_norm);
            fprintf('RMS tracking error              : %.6e\n', rms_tracking_error);
            fprintf('final ||theta_tilde(tf)||       : %.6e\n', final_parameter_error_norm);
            fprintf('Estimated nonzeros by threshold:\n');
            for k = 1:size(threshold_table, 1)
                fprintf('  tol = %.1e -> %d\n', threshold_table(k,1), threshold_table(k,2));
            end
            fprintf('============================\n\n');
        end

        function savedata(self, filedir)
            self.assertSimulationAvailable();
            self.ensureDirectory(filedir);

            t = self.sol.t(:);
            y = self.sol.y;
            idx = self.stateIndices();

            x = y(idx.x, :);
            xd = self.referenceTrajectory(t.');
            e = x - xd;
            tracking_error_norm = vecnorm(e, 2, 1);

            theta_hat = y(idx.theta_hat, :);
            theta_error = theta_hat - self.P.theta;
            theta_error_norm = vecnorm(theta_error, 2, 1);
            theta_hat_final = theta_hat(:, end);

            [support_metrics, threshold_table] = ...
                self.computeSupportMetrics(self.P.theta, theta_hat_final);

            coeff_table = self.coefficientTable(self.P.theta, theta_hat_final);
            paper_table = self.paperCoefficientTable(self.P.theta, theta_hat_final);

            full_state_data = [t, y.'];
            reference_data = [t, xd.'];
            trajectory_error_data = [t, e.'];
            tracking_error_norm_data = [t, tracking_error_norm.'];
            parameter_estimation_error_norm_data = [t, theta_error_norm.'];
            theta_hat_data = [t, theta_hat.'];

            max_points = 500;

            full_state_data_ds = self.downsampleTimeSeries(full_state_data, max_points);
            reference_data_ds = self.downsampleTimeSeries(reference_data, max_points);
            trajectory_error_data_ds = self.downsampleTimeSeries(trajectory_error_data, max_points);
            tracking_error_norm_data_ds = self.downsampleTimeSeries(tracking_error_norm_data, max_points);
            parameter_estimation_error_norm_data_ds = ...
                self.downsampleTimeSeries(parameter_estimation_error_norm_data, max_points);
            theta_hat_data_ds = self.downsampleTimeSeries(theta_hat_data, max_points);

            writematrix(full_state_data_ds, ...
                fullfile(filedir, 'full_state_data.dat'), ...
                'Delimiter', 'tab');

            writematrix(reference_data_ds, ...
                fullfile(filedir, 'reference_trajectory.dat'), ...
                'Delimiter', 'tab');

            writematrix(trajectory_error_data_ds, ...
                fullfile(filedir, 'trajectory_error_data.dat'), ...
                'Delimiter', 'tab');

            writematrix(tracking_error_norm_data_ds, ...
                fullfile(filedir, 'tracking_error_norm.dat'), ...
                'Delimiter', 'tab');

            writematrix(parameter_estimation_error_norm_data_ds, ...
                fullfile(filedir, 'parameter_estimation_error_norm.dat'), ...
                'Delimiter', 'tab');

            writematrix(theta_hat_data_ds, ...
                fullfile(filedir, 'theta_hat_data.dat'), ...
                'Delimiter', 'tab');

            writematrix(threshold_table, ...
                fullfile(filedir, 'threshold_nonzero_counts.dat'), ...
                'Delimiter', 'tab');

            writetable(coeff_table, fullfile(filedir, 'coefficient_table.csv'));
            writetable(paper_table, fullfile(filedir, 'paper_coefficient_table.csv'));

            summary = struct();
            summary.lambda = self.P.sparsity_lambda;
            summary.kappa = self.P.kappa;
            summary.k = self.P.k;
            summary.k_theta = self.P.k_theta;
            summary.K_c = self.P.K_c;
            summary.gamma = self.P.gamma;
            summary.T = self.P.T;
            summary.t_star = self.P.t_star;
            summary.t_f = self.P.t_f;
            summary.sparsity_tol = self.P.sparsity_tol;
            summary.rank_tol = self.P.rank_tol;
            summary.final_tracking_error_norm = norm(e(:, end));

            if numel(t) > 1 && (t(end) > t(1))
                summary.rms_tracking_error = sqrt(trapz(t, sum(e.^2,1).') / (t(end) - t(1)));
            else
                summary.rms_tracking_error = norm(e(:, end));
            end

            summary.final_parameter_error_norm = norm(theta_error(:, end));
            summary.true_nonzeros = support_metrics.true_nonzeros;
            summary.estimated_nonzeros = support_metrics.estimated_nonzeros;
            summary.tp = support_metrics.tp;
            summary.fp = support_metrics.fp;
            summary.fn = support_metrics.fn;
            summary.tn = support_metrics.tn;
            summary.threshold_table = threshold_table;
            summary.theta_true = self.P.theta;
            summary.theta_hat_final = theta_hat_final;
            summary.max_saved_points = max_points;
            summary.original_num_time_points = numel(t);
            summary.saved_num_tracking_error_points = size(tracking_error_norm_data_ds, 1);
            summary.saved_num_parameter_error_points = size(parameter_estimation_error_norm_data_ds, 1);

            save(fullfile(filedir, 'summary.mat'), 'summary');
        end

        function results = runLambdaSweep(self, lambda_values, output_dir, varargin)
            if nargin < 3 || isempty(output_dir)
                output_dir = fullfile(pwd, 'results_lambda_sweep');
            end

            opts = self.parseSweepOptions(varargin{:});
            self.ensureDirectory(output_dir);

            num_cases = numel(lambda_values);
            results = repmat(struct(), num_cases, 1);
            summary_rows = zeros(num_cases, 10);

            original_lambda = self.P.sparsity_lambda;
            cleanupObj = onCleanup(@() self.restoreLambda(original_lambda)); %#ok<NASGU>

            for k = 1:num_cases
                lambda_k = lambda_values(k);
                self.P.sparsity_lambda = lambda_k;
                self.lambda = lambda_k;

                case_name = self.lambdaLabel(lambda_k);
                case_dir = fullfile(output_dir, case_name);
                data_dir = fullfile(case_dir, 'data');
                fig_dir = fullfile(case_dir, 'figures');

                self.ensureDirectory(case_dir);
                self.ensureDirectory(data_dir);
                self.ensureDirectory(fig_dir);

                fprintf('\nRunning case %d/%d with lambda = %.6g\n', ...
                    k, num_cases, lambda_k);

                self.simulate();

                metrics = self.plotsimulations( ...
                    'Visible', opts.Visible, ...
                    'SaveFigures', true, ...
                    'SaveAppendixFigures', opts.SaveAppendixFigures, ...
                    'FigureDir', fig_dir);

                self.savedata(data_dir);

                results(k).lambda = lambda_k;
                results(k).case_dir = case_dir;
                results(k).data_dir = data_dir;
                results(k).figure_dir = fig_dir;
                results(k).metrics = metrics;

                summary_rows(k,:) = [ ...
                    lambda_k, ...
                    metrics.final_tracking_error_norm, ...
                    metrics.rms_tracking_error, ...
                    metrics.final_parameter_error_norm, ...
                    metrics.true_nonzeros, ...
                    metrics.estimated_nonzeros, ...
                    metrics.tp, ...
                    metrics.fp, ...
                    metrics.fn, ...
                    metrics.tn];
            end

            summary_table = array2table(summary_rows, ...
                'VariableNames', { ...
                'lambda', ...
                'final_tracking_error_norm', ...
                'rms_tracking_error', ...
                'final_parameter_error_norm', ...
                'true_nonzeros', ...
                'estimated_nonzeros', ...
                'TP', ...
                'FP', ...
                'FN', ...
                'TN'});

            paper_summary_table = summary_table(:, { ...
                'lambda', ...
                'estimated_nonzeros', ...
                'TP', ...
                'FP', ...
                'FN', ...
                'rms_tracking_error', ...
                'final_parameter_error_norm'});

            writetable(summary_table, ...
                fullfile(output_dir, 'lambda_sweep_summary.csv'));

            writetable(paper_summary_table, ...
                fullfile(output_dir, 'paper_lambda_sweep_summary.csv'));

            save(fullfile(output_dir, 'lambda_sweep_results.mat'), ...
                'results', 'lambda_values', 'summary_table', 'paper_summary_table');

            self.plotLambdaSweepSummary(results, output_dir, opts.Visible);
        end

        function data_ds = downsampleTimeSeries(self, data, max_points) %#ok<INUSD>
            if isempty(data)
                data_ds = data;
                return;
            end

            n_rows = size(data, 1);
            if n_rows <= max_points
                data_ds = data;
                return;
            end

            idx = self.downsampleIndices(n_rows, max_points);
            data_ds = data(idx, :);
        end

        function idx = downsampleIndices(self, n_rows, max_points) %#ok<INUSD>
            if n_rows <= max_points
                idx = (1:n_rows).';
                return;
            end

            idx = unique(round(linspace(1, n_rows, max_points))).';
            idx(1) = 1;
            idx(end) = n_rows;
        end

        function plotLambdaSweepSummary(self, results, output_dir, visible)
            if nargin < 4
                visible = true;
            end

            self.ensureDirectory(output_dir);

            if visible
                fig_visible = 'on';
            else
                fig_visible = 'off';
            end

            lambda = arrayfun(@(s) s.lambda, results);
            rms_e = arrayfun(@(s) s.metrics.rms_tracking_error, results);
            final_theta = arrayfun(@(s) s.metrics.final_parameter_error_norm, results);
            est_nz = arrayfun(@(s) s.metrics.estimated_nonzeros, results);
            fp = arrayfun(@(s) s.metrics.fp, results);
            tp = arrayfun(@(s) s.metrics.tp, results);

            xplot = lambda;
            if any(lambda > 0)
                min_positive = min(lambda(lambda > 0));
                xplot(lambda == 0) = min_positive / 2;
            else
                xplot(:) = 1;
            end

            figure_handles = gobjects(3,1);
            figure_names = { ...
                'paper_lambda_vs_rms_tracking_error', ...
                'paper_lambda_vs_parameter_error', ...
                'paper_lambda_vs_support_counts'};

            figure_handles(1) = figure('Visible', fig_visible);
            clf;
            semilogx(xplot, rms_e, '-o', 'LineWidth', 1.5);
            xlabel('\lambda');
            ylabel('RMS tracking error');
            title('RMS Tracking Error vs Sparsity Weight');
            grid minor;

            figure_handles(2) = figure('Visible', fig_visible);
            clf;
            semilogx(xplot, final_theta, '-o', 'LineWidth', 1.5);
            xlabel('\lambda');
            ylabel('final ||\theta~(t_f)||');
            title('Final Parameter Error vs Sparsity Weight');
            grid minor;

            figure_handles(3) = figure('Visible', fig_visible);
            clf;
            semilogx(xplot, est_nz, '-o', 'LineWidth', 1.5, ...
                'DisplayName', 'Estimated nonzeros');
            hold on;
            semilogx(xplot, tp, '-d', 'LineWidth', 1.5, ...
                'DisplayName', 'True positives');
            semilogx(xplot, fp, '-s', 'LineWidth', 1.5, ...
                'DisplayName', 'False positives');
            hold off;
            xlabel('\lambda');
            ylabel('Count');
            title('Support Recovery vs Sparsity Weight');
            legend('Location', 'best');
            grid minor;

            self.saveFigures(figure_handles, figure_names, output_dir);
        end
    end

    methods (Access = private)
        function P = defaultParameters(self) %#ok<MANU>
            P = struct();

            % System dimensions
            P.n = 2;
            P.m = 2;
            P.N = 20;
            P.p = 20;

            % System parameter
            P.mu = 1.0;

            % True sparse coefficient vector
            P.theta = [ ...
                0; -1; -1; 0; 0; 0; 0; 0; 0; 0; ...
                0; -0.5; -0.5; 0; 0; 0; 0; -0.5; 0; 0];

            % Control / adaptation gains
            P.k = 10;
            P.k_theta = 1;
            P.lambda_star = 0.5;
            P.gamma = [];
            P.kappa = 0.01;
            P.K_c = 0.1;

            % Integration parameters
            P.T = 0.25;
            P.t_star = 0.05;
            P.t_f = 100;

            % Sparsity parameters
            P.sparsity_lambda = 1e-3;
            P.sparsity_tol = 0.05;

            % Rank logic
            P.rank_tol = 1e-6;

            % Solver tolerances
            P.rel_tol = 1e-3;
            P.abs_tol = 1e-4;

            % Initial conditions
            P.x_0 = [-1; 1];
            P.theta_hat_0 = [];
            P.support_thresholds = [1e-3; 1e-2; 5e-2; 1e-1];

            % Reference trajectory
            P.reference_mode = 'rich';
        end

        function P = applyOverrides(self, P, params)
            if isempty(params)
                return;
            end

            names = fieldnames(params);
            for k = 1:numel(names)
                P.(names{k}) = params.(names{k});
            end
        end

        function configureModel(self)
            phi = @(x) [ ...
                1, ...
                x(1), ...
                x(2), ...
                x(1)^2, ...
                x(1)*x(2), ...
                x(2)^2, ...
                x(1)^3, ...
                x(1)^2*x(2), ...
                x(1)*x(2)^2, ...
                x(2)^3];

            self.fun_Y = @(x) [phi(x),      zeros(1,10); ...
                               zeros(1,10), phi(x)];

            self.fun_g = @(x) eye(self.P.n);
            self.fun_f0 = @(x) [0; 0];
            self.fun_gplus = @(x) pinv(self.fun_g(x));

            switch lower(string(self.P.reference_mode))
                case "rich"
                    self.fun_xd = @(t) [ ...
                        sin(t) + 0.12*sin(3*t) - 0.04*sin(5*t); ...
                        0.95*sin(2*t) + 0.08*sin(4*t)];

                    self.fun_xd_dot = @(t) [ ...
                        cos(t) + 0.36*cos(3*t) - 0.20*cos(5*t); ...
                        1.90*cos(2*t) + 0.32*cos(4*t)];

                case "standard"
                    self.fun_xd = @(t) [sin(2*t); 2*cos(2*t)];
                    self.fun_xd_dot = @(t) [2*cos(2*t); -4*sin(2*t)];

                otherwise
                    error('ICLAffine:InvalidReferenceMode', ...
                        'reference_mode must be ''rich'' or ''standard''.');
            end

            self.fun_u = @(t,x,theta_hat) self.fun_gplus(x) * ( ...
                self.fun_xd_dot(t) ...
                - self.fun_Y(x) * theta_hat ...
                - self.fun_f0(x) ...
                + self.P.k * (self.fun_xd(t) - x));

            inferred_p = size(self.fun_Y(zeros(self.P.n,1)), 2);

            if isempty(self.P.p)
                self.P.p = inferred_p;
            end

            if isempty(self.P.gamma)
                self.P.gamma = 1 * eye(self.P.p);
            end

            if isempty(self.P.theta_hat_0)
                self.P.theta_hat_0 = zeros(self.P.p, 1);
            end
        end

        function validateParameters(self)
            n = self.P.n;
            p = self.P.p;

            Y0 = self.fun_Y(zeros(n,1));
            inferred_p = size(Y0, 2);

            if inferred_p ~= p
                error('ICLAffine:DimensionMismatch', ...
                    'P.p = %d but fun_Y(x) has %d columns.', p, inferred_p);
            end

            if ~isequal(size(self.P.theta), [p, 1])
                error('ICLAffine:DimensionMismatch', ...
                    'P.theta must have size [%d, 1].', p);
            end

            if ~isequal(size(self.P.theta_hat_0), [p, 1])
                error('ICLAffine:DimensionMismatch', ...
                    'P.theta_hat_0 must have size [%d, 1].', p);
            end

            if ~isequal(size(self.P.x_0), [n, 1])
                error('ICLAffine:DimensionMismatch', ...
                    'P.x_0 must have size [%d, 1].', n);
            end

            if ~isequal(size(self.P.gamma), [p, p])
                error('ICLAffine:DimensionMismatch', ...
                    'P.gamma must have size [%d, %d].', p, p);
            end

            if ~isscalar(self.P.k) || self.P.k <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.k must be a positive scalar.');
            end

            if ~isscalar(self.P.K_c) || self.P.K_c <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.K_c must be a positive scalar.');
            end

            if ~isscalar(self.P.sparsity_lambda) || self.P.sparsity_lambda < 0
                error('ICLAffine:InvalidParameter', ...
                    'P.sparsity_lambda must be a nonnegative scalar.');
            end

            if ~isscalar(self.P.sparsity_tol) || self.P.sparsity_tol <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.sparsity_tol must be a positive scalar.');
            end

            if ~isscalar(self.P.kappa) || self.P.kappa < 0
                error('ICLAffine:InvalidParameter', ...
                    'P.kappa must be a nonnegative scalar.');
            end

            if ~isscalar(self.P.rank_tol) || self.P.rank_tol <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.rank_tol must be a positive scalar.');
            end

            if ~isscalar(self.P.T) || self.P.T <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.T must be a positive scalar.');
            end

            if ~isscalar(self.P.t_f) || self.P.t_f <= 0
                error('ICLAffine:InvalidParameter', ...
                    'P.t_f must be a positive scalar.');
            end
        end

        function idx = stateIndices(self)
            n = self.P.n;
            p = self.P.p;

            idx.x = 1:n;
            idx.theta_hat = n + (1:p);
            idx.IY = n + p + (1:n*p);
            idx.Ifgu = n + p + n*p + (1:n);
        end

        function state_dot = ddefunContinuous(self, t, state, state_delayed) %#ok<INUSD>
            n = self.P.n;
            p = self.P.p;

            x = state(1:n,1);
            e = -self.fun_xd(t) + x;
            theta_hat = state(n+1:n+p,1);

            u = self.fun_u(t, x, theta_hat);

            I_Y_dot = self.fun_Y(x);
            I_fg_u_dot = self.fun_f0(x) + self.fun_g(x) * u;

            theta_hat_dot = ...
                self.P.gamma * self.fun_Y(x).' * e ...
                + self.P.gamma * self.P.k_theta * self.script_Y.' * ...
                    (self.script_X - self.script_FGu - self.script_Y * theta_hat) ...
                - self.P.gamma * self.P.k_theta * self.P.sparsity_lambda * sign(theta_hat);

            x_dot = self.fun_Y(x) * self.P.theta + self.fun_f0(x) + self.fun_g(x) * u;

            state_dot = [x_dot; theta_hat_dot; I_Y_dot(:); I_fg_u_dot];
        end

        function [value, isterminal, direction] = ddefunEvents(self, t, state, state_delayed)
            if t - self.tau >= self.P.t_star && t > self.P.T
                n = self.P.n;
                p = self.P.p;
                N = self.P.N;

                x = state(1:n,1);
                x_delayed = state_delayed(1:n,1);

                I_Y = reshape(state(n+p+1:n+p+n*p,1), n, p);
                I_Y_delayed = reshape(state_delayed(n+p+1:n+p+n*p,1), n, p);
                cal_Y = I_Y - I_Y_delayed;

                I_fg_u = state(n+p+n*p+1:n+p+n*p+n,1);
                I_fg_u_delayed = state_delayed(n+p+n*p+1:n+p+n*p+n,1);

                scaling_factor = sqrt(1 + self.P.kappa * norm(cal_Y, 'fro')^2);

                gram_current = self.script_Y.' * self.script_Y;
                gram_new = gram_current - self.script_Y(1:n,:).' * self.script_Y(1:n,:) ...
                    + (cal_Y.' * cal_Y) / (scaling_factor^2);

                candidate_lambda = min(eig((gram_new + gram_new.') / 2));

                if candidate_lambda >= self.P.lambda_star
                    self.script_Y(1:n*(N-1), :) = self.script_Y(n+1:n*N, :);
                    self.script_X(1:n*(N-1), 1) = self.script_X(n+1:n*N, 1);
                    self.script_FGu(1:n*(N-1), 1) = self.script_FGu(n+1:n*N, 1);

                    j = N;
                    row_idx = n*(j-1)+1:n*j;
                    self.script_Y(row_idx, :) = cal_Y / scaling_factor;
                    self.script_X(row_idx, 1) = (x - x_delayed) / scaling_factor;
                    self.script_FGu(row_idx, 1) = (I_fg_u - I_fg_u_delayed) / scaling_factor;

                    self.lambda = candidate_lambda;
                    value = 0;
                else
                    value = 1;
                end

                self.tau = t;
            else
                value = 1;
            end

            isterminal = 1;
            direction = 0;
        end

        function state = ddefunHistory(self, t, x0)
            if t < 0
                state = zeros(size(x0));
            else
                state = x0;
            end
        end

        function xd = referenceTrajectory(self, t)
            xd = zeros(self.P.n, numel(t));
            for k = 1:numel(t)
                xd(:, k) = self.fun_xd(t(k));
            end
        end

        function assertSimulationAvailable(self)
            if isempty(self.sol) || isempty(self.sol.t) || isempty(self.sol.y)
                error('ICLAffine:NoSimulationData', ...
                    'No simulation data found. Run simulate() first.');
            end
        end

        function ensureDirectory(~, dirpath)
            dirpath = char(string(dirpath));
            if ~exist(dirpath, 'dir')
                mkdir(dirpath);
            end
        end

        function saveFigures(~, fig_handles, fig_names, fig_dir)
            fig_dir = char(string(fig_dir));
            for k = 1:numel(fig_handles)
                savefig(fig_handles(k), fullfile(fig_dir, [fig_names{k}, '.fig']));
                exportgraphics(fig_handles(k), ...
                    fullfile(fig_dir, [fig_names{k}, '.png']), ...
                    'Resolution', 300);
            end
        end

        function label = lambdaLabel(~, lambda_value)
            label = sprintf('lambda_%s', strrep(sprintf('%.6g', lambda_value), '.', 'p'));
            label = strrep(label, '+', '');
            label = strrep(label, '-', 'm');
        end

        function opts = parsePlotOptions(~, varargin)
            parser = inputParser;
            addParameter(parser, 'Visible', true, @(x) islogical(x) && isscalar(x));
            addParameter(parser, 'SaveFigures', false, @(x) islogical(x) && isscalar(x));
            addParameter(parser, 'SaveAppendixFigures', true, @(x) islogical(x) && isscalar(x));
            addParameter(parser, 'FigureDir', '', @(x) ischar(x) || isstring(x));
            parse(parser, varargin{:});
            opts = parser.Results;
        end

        function opts = parseSweepOptions(~, varargin)
            parser = inputParser;
            addParameter(parser, 'Visible', false, @(x) islogical(x) && isscalar(x));
            addParameter(parser, 'SaveAppendixFigures', true, @(x) islogical(x) && isscalar(x));
            parse(parser, varargin{:});
            opts = parser.Results;
        end

        function restoreLambda(self, lambda_value)
            self.P.sparsity_lambda = lambda_value;
            self.lambda = lambda_value;
        end

        function [support_metrics, threshold_table] = ...
                computeSupportMetrics(self, theta_true, theta_hat_final)

            tol = self.P.sparsity_tol;
            true_support = abs(theta_true) > 1e-12;
            est_support = abs(theta_hat_final) > tol;

            support_metrics = struct();
            support_metrics.tp = sum(true_support & est_support);
            support_metrics.fp = sum(~true_support & est_support);
            support_metrics.fn = sum(true_support & ~est_support);
            support_metrics.tn = sum(~true_support & ~est_support);
            support_metrics.true_nonzeros = sum(true_support);
            support_metrics.estimated_nonzeros = sum(est_support);
            support_metrics.true_zeros = sum(~true_support);
            support_metrics.estimated_zeros = sum(~est_support);

            threshold_list = self.P.support_thresholds(:);
            threshold_table = zeros(numel(threshold_list), 2);
            for k = 1:numel(threshold_list)
                threshold_table(k,1) = threshold_list(k);
                threshold_table(k,2) = sum(abs(theta_hat_final) > threshold_list(k));
            end
        end

        function labels = libraryLabels(self) %#ok<MANU>
            base = {'1','x_1','x_2','x_1^2','x_1x_2','x_2^2','x_1^3','x_1^2x_2','x_1x_2^2','x_2^3'};
            labels = [strcat('\dot{x}_1:', base), strcat('\dot{x}_2:', base)];
        end

        function T = coefficientTable(self, theta_true, theta_hat_final)
            labels = self.libraryLabels();
            p = numel(theta_true);
            block = strings(p,1);
            block(1:p/2) = "\dot{x}_1";
            block(p/2+1:p) = "\dot{x}_2";

            T = table( ...
                (1:p).', ...
                block, ...
                string(labels(:)), ...
                theta_true(:), ...
                theta_hat_final(:), ...
                abs(theta_hat_final(:)), ...
                abs(theta_hat_final(:)) > self.P.sparsity_tol, ...
                'VariableNames', { ...
                'index', ...
                'equation', ...
                'term', ...
                'theta_true', ...
                'theta_hat_final', ...
                'abs_theta_hat_final', ...
                'active_at_tol'});
        end

        function T = paperCoefficientTable(self, theta_true, theta_hat_final)
            coeff_table = self.coefficientTable(theta_true, theta_hat_final);
            mask = abs(theta_true(:)) > 1e-12 | ...
                   abs(theta_hat_final(:)) > self.P.sparsity_tol;
            T = coeff_table(mask, :);
        end
    end
end