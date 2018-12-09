classdef RecurseMatrix < handle
    properties
        Gradient
        Time
        P
        rho
        S
        alpha
        sizeBatch
    end
    methods
        function obj = RecurseMatrix(grad,time,sizeBatch,alpha)
            obj.Gradient = grad;
            obj.Time = time;
            obj.P = eye(sizeBatch) ;%.* rand(sizeBatch,sizeBatch);%./sizeBatch;%randn(sizeBatch,sizeBatch);
            obj.S = eye(2);
            obj.rho = 100;
            obj.alpha = alpha;
            obj.sizeBatch = sizeBatch;
            
        end
        function obj = updateMatrix(obj,grad,time,PMat,SMat,newRho,newAlpha)
            obj.Gradient = grad;
            obj.Time = time;
            obj.P = PMat;%.* rand(sizeBatch,sizeBatch);%./sizeBatch;%randn(sizeBatch,sizeBatch);
            obj.S = SMat;
            obj.rho = newRho;
            obj.alpha = newAlpha;
            
        end
    end
end