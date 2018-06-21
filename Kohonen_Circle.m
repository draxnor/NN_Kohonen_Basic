% SOM; Kohonen's Neural Network
% Identifying circle
% Comments in Polish
% Author: Pawe³ Mêdyk

% Sieæ Kohonena 
% Rozpoznanie okregu
% Imiê i nazwisko autora: Pawe³ Mêdyk

close all;

P_vec=[100 25 10 100];  % wektor liczby probek danych
K_vec=[25 100 100 10];  % wektor liczby neuronów

for ind=1:4
    figure(1); hold on;
    % wyznacz okrag
    P=P_vec(ind);
    r=4; t=linspace(0,2*pi,P);
    x1= r*cos(t); x2=r*sin(t);
    
    % dane wejsciowe, liczba wejsc
    x=[x1;x2];      % macierz danych uczacych
    N=size(x,1);    % liczba przykladow uczacych
    K=K_vec(ind);   % liczba neuronow w sieci
    
    kol=sqrt(K);    % liczba kolumn mapy 2D
    C=zeros(K,2);   % wspolrzedne neuronów na mapie 2D
    for k=1:K
        C(k,:)=[floor((k-1)/kol)+1, mod(k-1,kol)+1]; % nadanie wspolrzednych na mapie
    end  % rozmieszczenie neuronow w przestrzeni
    % siec o strukturze 2D - kwadratowa (prostokat o rownych
    % odleglosciach miedzy punktami) dist(n(i),n(i+1))=1 dla kazdego
    % neuronu; jeden wiersz zawiera liczbê neuronów równ¹ 'kol'
    % n1       n2          n3          n4 ...
    % n(kol+1) n(kol+2)    n(kol+3)    n(kol+4) ...
    %%%%%
    %struktura sieci wagi i
    a=0; b=1;            % przedzial inicjowanych wag
    W=(b-a)*rand(N,K)+a; % inicjuj macierz wag
    
    subplot(4,2,2*ind-1);
    plot(W(1,:),W(2,:),'.b')
    xlabel('x'); ylabel('y');
    str_title=sprintf('Okrag. Wezly przed procesem uczenia. K=%d',K);
    title(str_title)
    
    dist = @(v1,v2) sqrt(sum((v2-v1).^2)); % odleglosc - norma euklidesowa
    %neighbour_f= @(d,lam) (d<lam).*1;          % f. sasiedztwa prostokatna
    neighbour_f= @(d,lam) exp(-d^2/(2*lam^2)); % f. sasiedztwa Gaussa
    Epoki=10000;% liczba epok
    w=1/Epoki;  % czestotliwosc zmian w 1 epoce
    alpha=0.8;  % max wartosc wspó³czynnika uczenia sie
    lambda=3;   % max promien sasiedztwa
    D=zeros(K,1);   % wektor odleglosci w przestrzeni danych wejsciowych
    for ep=1:Epoki
        L=randi([1 P],1); % wylosuj dane ucz.
        
        for k=1:K;  %odl neuronow od wejsc
            D(k)= dist(x(:,L),W(:,k)); % D(k) odleglosc k'tego neuronu od przykladu uczacego
        end
        % odszukaj zwyciezce, neuron o najwiekszym pobudzeniu, najblizej
        % wejscia; w - winner, w_i - winner index
        [winn,w_i]=min(D);
        % dla strategii WTA - popraw zwyciezce
        % W(:,w_i)= (W(:,w_i)+alpha*x(:,L))/(dist(0,W(:,w_i)+alpha*x(:,L))); % m.addytywna
        % W(:,w_i)= W(:,w_i) + alpha(x(:,L)-W(:,w_i));                       % m.subtraktywna
        
        % dla strategii WTM popraw zwyciezce
        % obliczenie odleglosci wszystkich neur. od zwyciezcy
        % aktualizacja wag wszystkich neuronow
        for k=1:K;
            map_k_dist=dist(C(w_i,:),C(k,:)); % odleglosc k'tego neuronu na mapie od zwyciezcy
            W(:,k)= W(:,k)+alpha*neighbour_f(map_k_dist,lambda)*(x(:,L)-W(:,k)); % m.subtraktywna
        end
        % redukcja parametrow
        alpha=(1-w)*alpha;
        lambda=(1-w)*lambda;
    end % zakonczenie cyklu
    
    % wykresy wplywu stosunku ilosci danych pomiarowych do liczby neuronow
    subplot(4,2,2*ind);
    plot(W(1,:),W(2,:),'.b')
    str_title=sprintf('Okrag. Wezly po procesie uczenia. K=%d',K);
    xlabel('x'); ylabel('y');
    title(str_title)
end

% wykres przedstawiajacy oryginalny okrag (dane uczace)
figure(2); hold on;
plot(x(1,:),x(2,:), '.b');
xlabel('x'); ylabel('y');
title('Okrag. Dane uczace. Siec Kohonena')