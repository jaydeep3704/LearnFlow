import { AuthDesign } from "@/components/general/AuthDesign";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import GoogleIcon from "@/assets/icons/google.png"
import GithubIcon from "@/assets/icons/github.png"
import Image from "next/image";

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
                    <CardContent className="space-y-4">
                        <Button variant={"secondary"} className="w-full py-5">
                            <Image src={GoogleIcon} alt="google-icon" className="size-5"/>
                             Login with Google
                        </Button>
                        <Button variant={"secondary"} className="w-full py-5">
                            <Image src={GithubIcon} alt="google-icon" className="size-5"/>
                             Login with Github
                        </Button>
                    </CardContent>
                </Card>
                
            </div>
        </div>
    )
}