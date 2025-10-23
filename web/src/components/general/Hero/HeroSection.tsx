import { SparklesIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { HeroImg } from "./HeroImg";

export function HeroSection(){
    return(
        <div className="px-[4%] lg:py-20 py-10">
            <div className="max-w-7xl mx-auto grid lg:grid-cols-5">
                <div className="lg:col-span-2 space-y-4">
                    <h1 className="lg:text-6xl md:text-5xl text-4xl lg:text-left text-center font-semibold ">
                        Transform YouTube Videos into {" "}
                        <span className="text-primary">
                        Structured Courses
                        </span>
                    </h1>
                    <p className="text-muted-foreground lg:text-2xl lg:text-left text-center text-balance">
                        AI-powered learning platform that automatically creates chapters, quizzes, and interactive experiences from any YouTube video.
                    </p>
                    <div className="flex gap-4 items-center justify-center lg:justify-start">
                        <Button >
                            <SparklesIcon/>
                            Start Learning For Free
                        </Button>
                        <Button variant={"outline"} >
                            View Demo
                        </Button>
                    </div>
                </div>
                <div className="col-span-3 hidden lg:block">
                    <HeroImg/>
                </div>
            </div>
        </div>
    )
}




