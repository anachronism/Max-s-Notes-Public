function [actionID, fitObserv, fitParams] = findTheoreticalBestAction(actionTable, esN0, FERCurves, weightVector, ModList, CodList, ModCodList)

    maxBER = 1e-6;
	RsMax = max(actionTable(:,1));
	RsMin = min(actionTable(:,1));
	BWMin = RsMin*(1+min(actionTable(:,4)));
	BWMax = RsMax*(1+max(actionTable(:,4)));
	TMax = RsMax*log2(max(actionTable(:,5)))*max(actionTable(:,6));
	TMin = RsMin*log2(min(actionTable(:,5)))*min(actionTable(:,6));
	EsMinLin = 10.0^(min(actionTable(:,2))/10);
	EsMaxLin = 10.0^(max(actionTable(:,2))/10);
	PConsumMinLin = EsMinLin*RsMin;
	PConsumMaxLin = EsMaxLin*RsMax;
	SpectEffMin = log2(min(actionTable(:,5)))*min(actionTable(:,6))/(1+max(actionTable(:,4)));
	SpectEffMax = log2(max(actionTable(:,5)))*max(actionTable(:,6))/(1+min(actionTable(:,4)));
	PEffMaxLog10 = log10(log2(max(actionTable(:,5)))*max(actionTable(:,6))/(EsMinLin*RsMin));
	PEffMinLog10 = log10(log2(min(actionTable(:,5)))*min(actionTable(:,6))/(EsMaxLin*RsMax));
	berDBMax = -10*log10(maxBER);
	berDBMin = -10*log10(1);
    
    Rs = actionTable(:,1);        
    esAdd = actionTable(:,2);
    roll_off = actionTable(:,4);
    M = actionTable(:,5);
    rate = actionTable(:,6);        

    measuredPowConsumedLin = 10.0.^(esAdd/10.0).*Rs;
    measuredPowConsumedLinComplement = PConsumMaxLin+PConsumMinLin - measuredPowConsumedLin;
    measuredPowEfficiencyLog10 = log10((log2(M).*rate)./measuredPowConsumedLin);
    measuredBandwidth = Rs.*(1.0+roll_off);
    measuredThroughput = Rs.*log2(M).*rate;
    measuredSpectralEff = log2(M).*rate./(1.0+roll_off);
    measuredBEREst = estimateBER(esN0,M,rate,FERCurves,ModList, CodList, ModCodList);
    measuredBEREstdB = -10.0.*log10(measuredBEREst);

    %populate observed params, normalized to [0,1]
    fitObservedParams = zeros(size(actionTable,1),6);
    fitObservedParams(:,1) = (measuredThroughput-TMin)/(TMax-TMin);
    fitObservedParams(:,2) = (measuredBEREstdB-berDBMin)/(berDBMax-berDBMin);
    fitObservedParams(:,3) = (measuredBandwidth-BWMin)/(BWMax-BWMin);
    fitObservedParams(:,4) = (measuredSpectralEff-SpectEffMin)/(SpectEffMax-SpectEffMin);
    fitObservedParams(:,5) = (measuredPowEfficiencyLog10-PEffMinLog10)/(PEffMaxLog10-PEffMinLog10);
    fitObservedParams(:,6) = (measuredPowConsumedLinComplement-PConsumMinLin)/(PConsumMaxLin-PConsumMinLin);
    
    fitObserved = sum(fitObservedParams .* repmat(weightVector,size(actionTable,1),1),2);
    fitObserved = (fitObservedParams(:,2)~=0.0).*fitObserved; %reality check
    
    [fitObserv,actionID] = max(fitObserved);
    fitParams = fitObservedParams(actionID);
    
end

function FERlin = estimateBER(EsN0dB,M,rate,FERCurves,ModList,CodList,ModCodList)
    FERlog10 = zeros(size(M,1),1);
    FERlin = zeros(size(M,1),1);

	%find modcod
    Modcod = zeros(size(M,1),1);
    for i=1:size(M,1)
        Modcod(i) = ModCodList((abs(ModList-M(i))<0.0001) & ((abs(CodList-rate(i))<0.0001)));
    end

	%find closest bounding points on curve
	minPoint = zeros(size(Modcod,1),1);
	maxPoint = zeros(size(Modcod,1),1);
    
    for j=1:size(Modcod,1)
        modcodChosenIdx = find(ModCodList == Modcod(j));
        for i=1:size(FERCurves{modcodChosenIdx,1},2)
            if(EsN0dB>=FERCurves{modcodChosenIdx,1}(i))
                minPoint(j) = i;
            end
            if(EsN0dB < FERCurves{modcodChosenIdx,1}(i) && maxPoint(j)==0)
                maxPoint(j) = i;
            end
        end
        
        if(minPoint(j)==0) %to the left of the fer curve
            minPoint(j) = maxPoint(j); %use first two points and linearly interpolate leftward
            maxPoint(j) = maxPoint(j)+1;
        end
        if(maxPoint(j)==0)  %to the right of the fer cruve
            maxPoint(j) = minPoint(j); %use last two points and linearly interpolate rightward
            minPoint(j) = minPoint(j)-1;
        end

        FERlog10(j) = ((log10(FERCurves{modcodChosenIdx,2}(maxPoint(j)))-log10(FERCurves{modcodChosenIdx,2}(minPoint(j))))/(FERCurves{modcodChosenIdx,1}(maxPoint(j))-FERCurves{modcodChosenIdx,1}(minPoint(j)))) ...
                    *(EsN0dB-FERCurves{modcodChosenIdx,1}(minPoint(j)))+log10(FERCurves{modcodChosenIdx,2}(minPoint(j)));
                
        %hard limit at 0 and -12
        if(FERlog10(j)>=0)
            FERlog10(j)=0;
        end
        if(FERlog10(j)<-6)
            FERlog10(j) = -6;
        end
        %convert to linear
        FERlin(j) = 10^FERlog10(j);

    end


end