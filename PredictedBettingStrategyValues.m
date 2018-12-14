clc; close all;

% Edges vector
edges = -59.5:1:59.5;

% Imported from python
cd = load('CountDistribution.mat');
count = cd.counts;

% Plot the histigram of given data
figure
[N] = histogram(count, edges, 'Normalization', 'pdf');
xlabel('Count')
ylabel('Normalized Frequency')
title('Fitting Normal Distribution to Count Frequency');
hold on

% Pull histragram data off the chart
ydata = N.BinCounts/sum(N.BinCounts);
xdata = edges(1:end-1) + 0.5;

% Do a least squares fitting of data
fun = @(q) leastSquares(xdata, ydata, 0, q);
q_guess = [3];
LS_parameter = fminsearch(fun, q_guess);

% Cauchy pdf to fit the distribution, normal was not doing hot
cauchy_approximation = cauchypdf(xdata, 0, LS_parameter);
plot(xdata, cauchy_approximation, 'LineWidth', 1.5)'
xlim([-25, 25])

%% Part 2, use pdf to margionalize out the count from expected value to find E[x | pibet]
betpolicy =    [.5, 1, 5;
                0, 1, 5;
                0, 0, 5;
                1, 1, 1;
                .5,1, 2.5];
betlevel = [0, 2.5];
EV = zeros(size(betpolicy,1), 1);
bet = zeros(1, length(xdata));

structure = load('startVCount.mat');
Vloaded = structure.CountValues;
loadCount = -15:1:15;
Vstart = zeros(1, length(xdata));

figure
hold on
ax1 = gca;
figure 
hold on
ax2=gca;
chart = 1;
for i = 1:size(betpolicy, 1)
for count = min(xdata):1:max(xdata)
    if count > 15
        Vstart(count + max(xdata) + 1) = Vloaded(end);
    elseif count < -15
        Vstart(count + max(xdata) + 1) = Vloaded(1);
    else
        index = find(loadCount == count);
        Vstart(count + max(xdata) + 1) = Vloaded(index);
    end
        
    if count < betlevel(1)
        bet(count + max(xdata) + 1) = betpolicy(i, 1);
    elseif count < betlevel(2)
        bet(count + max(xdata) + 1) = betpolicy(i, 2);
    else
        bet(count + max(xdata) + 1) = betpolicy(i, 3);
    end
end

subplot(5,2,chart)
h = area(xdata, bet, 'LineWidth', 1, 'FaceColor', [0 17 0]/30);
h.FaceAlpha = 0.2;
grid on
xlabel('True Count')
ylabel('Bet (units)')
title(strcat('Betting Policy for ', '[', num2str(betpolicy(i, 1)), ' ', num2str(betpolicy(i,2)), ' ', num2str(betpolicy(i,3)), ']'))
set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on');
ylim([0, max(bet) + .5])
xlim([-20, 20])
chart = chart+1;

EV(i) = trapz(xdata, cauchy_approximation.*bet.*Vstart);
disp('Expected Value is:')
disp(EV(i))
disp('For Bet Policy:')
disp(betpolicy(i, :))

subplot(5, 2, chart)
h = area(xdata, cauchy_approximation.*bet.*Vstart, 'LineWidth', 1, 'FaceColor', [0 17 0]/30);
h.FaceAlpha = 0.2;
grid('on')
xlabel('True Count')
ylabel('Deaggregation')
title('Deaggregation of Expected Value')
set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on');
xlim([-40, 40])
chart = chart + 1;
end

figure
plot(xdata, Vstart, 'LineWidth', 1)
grid on
xlabel('True Count')
ylabel('Deag')
set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on');
% xlim([-15 15])
grid on
title('Expected Value Given True Card Count')
%% Save Exact Values Calculated Explicitly using probability
% use distributions given via similuate, expected values for count
% Given by the MDP and value iteration, check this probablistic value with
% model free monte carlo simulations for the expected value, ensure optimum
% policy is correct
save('EV_Exact', 'EV', 'betpolicy')
