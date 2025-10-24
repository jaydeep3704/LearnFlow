"use client"
import { useFormStatus } from "react-dom";
import { Button } from "../ui/button";
import {  Loader2 } from "lucide-react";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface GeneralSubmitButtonProps{
    text:string;
    variant?: "link" | "default" | "destructive" | "outline" | "secondary" | "ghost" | null | undefined;
    width?:string;
    icon?:ReactNode;
    error?:string;
    success?:string;
    className?:string
}

export function GeneralSubmitButton({text,variant,width,icon,error,success,className}:GeneralSubmitButtonProps){
    const {pending}=useFormStatus()

    return(
        <Button className={cn(width,className)} variant={variant} disabled={pending} >
            {pending ? (
                <>
                 <Loader2 className="size-4 animate-spin"/>
                </>
            ) : (
                <>
                 {icon && <div>{icon}</div>} 
                 <span>{text}</span>
                </>
            )}
        </Button>
    )
}

