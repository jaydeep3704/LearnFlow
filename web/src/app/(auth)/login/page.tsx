
import { AuthDesign } from "@/components/general/AuthDesign";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import { Loader2 } from "lucide-react";
import { LoginForm } from "@/components/forms/LoginForm";


export default function Login(){
    


    return(
        <div className="lg:grid grid-cols-7 h-screen w-full ">
            <div className="col-span-3 hidden lg:block">
                <AuthDesign/> 
            </div>
            <div className="col-span-4 lg:py-0 lg:px-0 py-[4%] px-[4%] flex justify-center items-center">
                <Card className="lg:w-[70%] w-full">
                    <CardHeader className="">
                      <CardTitle className="lg:text-2xl  text-lg text-center">Welcome To LearnFlow</CardTitle>
                      <CardDescription className="text-center text-lg text-muted-foreground">Login with github or google account</CardDescription>
                    </CardHeader>
                    <LoginForm/>
                    
                </Card>
                
            </div>
        </div>
    )
}

